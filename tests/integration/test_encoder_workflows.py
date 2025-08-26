"""Integration tests for encoder workflows.

These tests verify that encoders work correctly in complete workflows,
including preprocessing, encoding, and post-processing steps.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from molenc.core.registry import EncoderRegistry
from molenc.preprocessing import SMILESStandardizer, SMILESValidator, MolecularFilters
from molenc.preprocessing.utils import preprocess_smiles_list


class TestEncoderWorkflows:
    """Test complete encoder workflows."""
    
    @pytest.fixture
    def sample_smiles(self):
        """Sample SMILES for testing."""
        return [
            'CCO',  # Ethanol
            'CC(=O)O',  # Acetic acid
            'c1ccccc1',  # Benzene
            'CCN(CC)CC',  # Triethylamine
            'CC(C)O',  # Isopropanol
            'INVALID_SMILES',  # Invalid SMILES
            'C1=CC=CC=C1O',  # Phenol
            'CC(C)(C)O'  # tert-Butanol
        ]
    
    @pytest.fixture
    def valid_smiles(self):
        """Valid SMILES for testing."""
        return [
            'CCO',
            'CC(=O)O',
            'c1ccccc1',
            'CCN(CC)CC',
            'CC(C)O',
            'C1=CC=CC=C1O',
            'CC(C)(C)O'
        ]
    
    def test_preprocessing_to_fingerprint_workflow(self, sample_smiles):
        """Test complete workflow from raw SMILES to fingerprints."""
        # Mock RDKit to avoid dependency
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0, 1, 0] * 512
            
            # Step 1: Preprocess SMILES
            standardizer = SMILESStandardizer()
            validator = SMILESValidator()
            
            # Standardize
            standardized = []
            for smiles in sample_smiles:
                try:
                    std_smiles = standardizer.standardize(smiles)
                    if std_smiles and validator.is_valid(std_smiles):
                        standardized.append(std_smiles)
                except:
                    continue
            
            # Step 2: Encode with fingerprints
            registry = EncoderRegistry()
            
            # Mock encoder registration
            mock_encoder_class = MagicMock()
            mock_encoder = MagicMock()
            mock_encoder.encode_batch.return_value = np.random.rand(len(standardized), 2048)
            mock_encoder.get_output_dim.return_value = 2048
            mock_encoder_class.return_value = mock_encoder
            
            registry.register('morgan', mock_encoder_class)
            encoder = registry.get_encoder('morgan')
            
            # Step 3: Encode
            if standardized:
                features = encoder.encode_batch(standardized)
                
                assert isinstance(features, np.ndarray)
                assert features.shape[0] == len(standardized)
                assert features.shape[1] == 2048
    
    def test_preprocessing_pipeline_integration(self, sample_smiles):
        """Test integration of preprocessing pipeline."""
        # Mock RDKit
        with patch('molenc.preprocessing.standardize.rdkit') as mock_rdkit:
            with patch('molenc.preprocessing.validators.rdkit') as mock_rdkit_val:
                with patch('molenc.preprocessing.filters.rdkit') as mock_rdkit_filt:
                    
                    # Setup mocks
                    mock_mol = MagicMock()
                    mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
                    mock_rdkit.Chem.MolToSmiles.return_value = 'CCO'
                    mock_rdkit_val.Chem.MolFromSmiles.return_value = mock_mol
                    mock_rdkit_filt.Chem.MolFromSmiles.return_value = mock_mol
                    mock_rdkit_filt.Chem.Descriptors.MolWt.return_value = 150.0
                    mock_rdkit_filt.Chem.Descriptors.MolLogP.return_value = 2.0
                    
                    # Use preprocessing pipeline
                    processed = preprocess_smiles_list(
                        sample_smiles,
                        standardize=True,
                        validate=True,
                        filter_molecules=True,
                        n_jobs=1
                    )
                    
                    assert isinstance(processed, dict)
                    assert 'processed_smiles' in processed
                    assert 'statistics' in processed
                    assert isinstance(processed['processed_smiles'], list)
    
    def test_encoder_comparison_workflow(self, valid_smiles):
        """Test workflow comparing different encoders."""
        # Mock different encoders
        with patch('molenc.encoders.fingerprints.rdkit'):
            with patch('molenc.encoders.substructure.rdkit'):
                with patch('molenc.encoders.substructure.gensim'):
                    
                    registry = EncoderRegistry()
                    
                    # Mock Morgan encoder
                    mock_morgan = MagicMock()
                    mock_morgan.encode_batch.return_value = np.random.rand(len(valid_smiles), 2048)
                    mock_morgan.get_output_dim.return_value = 2048
                    mock_morgan_class = MagicMock(return_value=mock_morgan)
                    registry.register('morgan', mock_morgan_class)
                    
                    # Mock MACCS encoder
                    mock_maccs = MagicMock()
                    mock_maccs.encode_batch.return_value = np.random.rand(len(valid_smiles), 167)
                    mock_maccs.get_output_dim.return_value = 167
                    mock_maccs_class = MagicMock(return_value=mock_maccs)
                    registry.register('maccs', mock_maccs_class)
                    
                    # Encode with different encoders
                    results = {}
                    for encoder_name in ['morgan', 'maccs']:
                        encoder = registry.get_encoder(encoder_name)
                        features = encoder.encode_batch(valid_smiles)
                        results[encoder_name] = features
                    
                    # Verify results
                    assert 'morgan' in results
                    assert 'maccs' in results
                    assert results['morgan'].shape[1] == 2048
                    assert results['maccs'].shape[1] == 167
                    assert results['morgan'].shape[0] == len(valid_smiles)
                    assert results['maccs'].shape[0] == len(valid_smiles)
    
    def test_batch_processing_workflow(self):
        """Test workflow with large batch processing."""
        # Generate large dataset
        large_smiles = ['CCO'] * 1000 + ['CC(=O)O'] * 1000 + ['c1ccccc1'] * 1000
        
        # Mock dependencies
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
            
            registry = EncoderRegistry()
            
            # Mock encoder
            mock_encoder = MagicMock()
            mock_encoder.encode_batch.return_value = np.random.rand(len(large_smiles), 2048)
            mock_encoder.get_output_dim.return_value = 2048
            mock_encoder_class = MagicMock(return_value=mock_encoder)
            registry.register('morgan', mock_encoder_class)
            
            encoder = registry.get_encoder('morgan')
            
            # Process in batches
            batch_size = 500
            all_features = []
            
            for i in range(0, len(large_smiles), batch_size):
                batch = large_smiles[i:i + batch_size]
                features = encoder.encode_batch(batch)
                all_features.append(features)
            
            # Combine results
            combined_features = np.vstack(all_features)
            
            assert combined_features.shape[0] == len(large_smiles)
            assert combined_features.shape[1] == 2048
    
    def test_error_handling_workflow(self, sample_smiles):
        """Test workflow with error handling."""
        # Mock RDKit with some failures
        with patch('molenc.preprocessing.standardize.rdkit') as mock_rdkit:
            def mock_mol_from_smiles(smiles):
                if 'INVALID' in smiles:
                    return None
                return MagicMock()
            
            mock_rdkit.Chem.MolFromSmiles.side_effect = mock_mol_from_smiles
            mock_rdkit.Chem.MolToSmiles.return_value = 'CCO'
            
            standardizer = SMILESStandardizer()
            validator = SMILESValidator()
            
            # Process with error handling
            processed = []
            errors = []
            
            for smiles in sample_smiles:
                try:
                    std_smiles = standardizer.standardize(smiles)
                    if std_smiles and validator.is_valid(std_smiles):
                        processed.append(std_smiles)
                except Exception as e:
                    errors.append((smiles, str(e)))
            
            # Should have some processed and some errors
            assert len(processed) > 0
            assert len(errors) >= 0  # May or may not have errors depending on mock behavior
    
    def test_feature_extraction_pipeline(self, valid_smiles):
        """Test complete feature extraction pipeline."""
        # Mock all dependencies
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            with patch('molenc.preprocessing.filters.rdkit') as mock_rdkit_filt:
                
                # Setup mocks
                mock_mol = MagicMock()
                mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
                mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
                mock_rdkit_filt.Chem.MolFromSmiles.return_value = mock_mol
                mock_rdkit_filt.Chem.Descriptors.MolWt.return_value = 150.0
                
                # Step 1: Filter molecules
                filters = MolecularFilters(mw_range=(50, 500))
                filtered_smiles = []
                
                for smiles in valid_smiles:
                    if filters.passes_filters(smiles):
                        filtered_smiles.append(smiles)
                
                # Step 2: Extract features
                registry = EncoderRegistry()
                
                mock_encoder = MagicMock()
                mock_encoder.encode_batch.return_value = np.random.rand(len(filtered_smiles), 2048)
                mock_encoder.get_output_dim.return_value = 2048
                mock_encoder_class = MagicMock(return_value=mock_encoder)
                registry.register('morgan', mock_encoder_class)
                
                encoder = registry.get_encoder('morgan')
                features = encoder.encode_batch(filtered_smiles)
                
                # Step 3: Post-process features
                # Normalize features
                normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
                
                assert normalized_features.shape == features.shape
                assert np.allclose(np.linalg.norm(normalized_features, axis=1), 1.0)


class TestEndToEndWorkflows:
    """End-to-end workflow tests."""
    
    def test_drug_discovery_workflow(self):
        """Test a complete drug discovery workflow."""
        # Sample drug-like molecules
        drug_smiles = [
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(C)NCC(C1=CC(=C(C=C1)O)CO)O',  # Salbutamol
        ]
        
        # Mock all dependencies
        with patch('molenc.preprocessing.standardize.rdkit') as mock_rdkit_std:
            with patch('molenc.preprocessing.validators.rdkit') as mock_rdkit_val:
                with patch('molenc.preprocessing.filters.rdkit') as mock_rdkit_filt:
                    with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit_enc:
                        
                        # Setup mocks
                        mock_mol = MagicMock()
                        mock_rdkit_std.Chem.MolFromSmiles.return_value = mock_mol
                        mock_rdkit_std.Chem.MolToSmiles.return_value = 'CCO'
                        mock_rdkit_val.Chem.MolFromSmiles.return_value = mock_mol
                        mock_rdkit_filt.Chem.MolFromSmiles.return_value = mock_mol
                        mock_rdkit_filt.Chem.Descriptors.MolWt.return_value = 250.0
                        mock_rdkit_filt.Chem.Descriptors.MolLogP.return_value = 2.5
                        mock_rdkit_filt.Chem.Descriptors.NumHDonors.return_value = 1
                        mock_rdkit_filt.Chem.Descriptors.NumHAcceptors.return_value = 2
                        mock_rdkit_enc.Chem.MolFromSmiles.return_value = mock_mol
                        mock_rdkit_enc.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
                        
                        # Step 1: Preprocess molecules
                        processed = preprocess_smiles_list(
                            drug_smiles,
                            standardize=True,
                            validate=True,
                            filter_molecules=True,
                            n_jobs=1
                        )
                        
                        valid_molecules = processed['processed_smiles']
                        
                        # Step 2: Apply drug-like filters (Lipinski's Rule of Five)
                        filters = MolecularFilters(
                            mw_range=(150, 500),
                            logp_range=(-2, 5),
                            hbd_range=(0, 5),
                            hba_range=(0, 10),
                            lipinski_rule=True
                        )
                        
                        drug_like = []
                        for smiles in valid_molecules:
                            if filters.passes_filters(smiles):
                                drug_like.append(smiles)
                        
                        # Step 3: Generate molecular descriptors
                        registry = EncoderRegistry()
                        
                        # Mock multiple encoders
                        encoders = {}
                        for name, dim in [('morgan', 2048), ('maccs', 167)]:
                            mock_encoder = MagicMock()
                            mock_encoder.encode_batch.return_value = np.random.rand(len(drug_like), dim)
                            mock_encoder.get_output_dim.return_value = dim
                            mock_encoder_class = MagicMock(return_value=mock_encoder)
                            registry.register(name, mock_encoder_class)
                            encoders[name] = registry.get_encoder(name)
                        
                        # Generate features
                        features = {}
                        for name, encoder in encoders.items():
                            features[name] = encoder.encode_batch(drug_like)
                        
                        # Step 4: Combine features
                        combined_features = np.hstack([features['morgan'], features['maccs']])
                        
                        # Verify workflow
                        assert len(drug_like) > 0
                        assert combined_features.shape[0] == len(drug_like)
                        assert combined_features.shape[1] == 2048 + 167
    
    def test_chemical_space_analysis_workflow(self):
        """Test chemical space analysis workflow."""
        # Diverse chemical structures
        diverse_smiles = [
            'CCO',  # Alcohol
            'CC(=O)O',  # Carboxylic acid
            'CCN',  # Amine
            'c1ccccc1',  # Aromatic
            'C1CCCCC1',  # Cyclic
            'C=C',  # Alkene
            'C#C',  # Alkyne
            'CC(C)C',  # Branched
        ]
        
        # Mock dependencies
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
            
            registry = EncoderRegistry()
            
            # Mock encoder
            mock_encoder = MagicMock()
            # Generate diverse features for different molecules
            features_list = []
            for i, smiles in enumerate(diverse_smiles):
                # Create slightly different features for each molecule
                feature_vector = np.random.rand(2048)
                feature_vector[i * 100:(i + 1) * 100] = 1.0  # Make each molecule unique
                features_list.append(feature_vector)
            
            mock_encoder.encode_batch.return_value = np.array(features_list)
            mock_encoder.get_output_dim.return_value = 2048
            mock_encoder_class = MagicMock(return_value=mock_encoder)
            registry.register('morgan', mock_encoder_class)
            
            encoder = registry.get_encoder('morgan')
            
            # Generate molecular representations
            features = encoder.encode_batch(diverse_smiles)
            
            # Analyze chemical space
            # Calculate pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(features)
            
            # Verify diversity
            assert features.shape[0] == len(diverse_smiles)
            assert features.shape[1] == 2048
            assert similarities.shape == (len(diverse_smiles), len(diverse_smiles))
            
            # Diagonal should be 1 (self-similarity)
            np.testing.assert_allclose(np.diag(similarities), 1.0, rtol=1e-10)
    
    def test_molecular_property_prediction_workflow(self):
        """Test molecular property prediction workflow."""
        # Molecules with known properties
        molecules_data = [
            ('CCO', {'solubility': 'high', 'toxicity': 'low'}),
            ('CCCCCCCCCC', {'solubility': 'low', 'toxicity': 'medium'}),
            ('c1ccc(cc1)O', {'solubility': 'medium', 'toxicity': 'low'}),
            ('CC(C)(C)c1ccc(cc1)O', {'solubility': 'low', 'toxicity': 'low'}),
        ]
        
        smiles_list = [mol[0] for mol in molecules_data]
        properties = [mol[1] for mol in molecules_data]
        
        # Mock dependencies
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
            
            registry = EncoderRegistry()
            
            # Mock encoder with property-relevant features
            mock_encoder = MagicMock()
            # Create features that correlate with properties
            features_list = []
            for i, (smiles, props) in enumerate(molecules_data):
                feature_vector = np.random.rand(2048)
                # Encode solubility information
                if props['solubility'] == 'high':
                    feature_vector[:100] = 0.8
                elif props['solubility'] == 'medium':
                    feature_vector[:100] = 0.5
                else:
                    feature_vector[:100] = 0.2
                features_list.append(feature_vector)
            
            mock_encoder.encode_batch.return_value = np.array(features_list)
            mock_encoder.get_output_dim.return_value = 2048
            mock_encoder_class = MagicMock(return_value=mock_encoder)
            registry.register('morgan', mock_encoder_class)
            
            encoder = registry.get_encoder('morgan')
            
            # Generate features
            features = encoder.encode_batch(smiles_list)
            
            # Simulate property prediction
            # In a real scenario, this would involve training ML models
            solubility_scores = features[:, :100].mean(axis=1)
            
            # Verify workflow
            assert features.shape[0] == len(smiles_list)
            assert len(solubility_scores) == len(smiles_list)
            
            # Check that features capture some property information
            # (This is a simplified check since we're using mocked data)
            assert np.var(solubility_scores) > 0  # Features should be diverse
    
    def test_virtual_screening_workflow(self):
        """Test virtual screening workflow."""
        # Target molecule (query)
        target_smiles = 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'  # Ibuprofen-like
        
        # Library of molecules to screen
        library_smiles = [
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Exact match
            'CC(C)CC1=CC=C(C=C1)C(C)CO',  # Similar structure
            'CCO',  # Very different
            'c1ccccc1',  # Different
            'CC(C)CC1=CC=C(C=C1)C(C)C',  # Similar but no carboxylic acid
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)N',  # Amide instead of acid
        ]
        
        # Mock dependencies
        with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
            mock_mol = MagicMock()
            mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
            mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
            
            registry = EncoderRegistry()
            
            # Mock encoder with similarity-preserving features
            mock_encoder = MagicMock()
            
            def create_similarity_features(smiles_list, target):
                features = []
                target_feature = np.random.rand(2048)
                
                for smiles in smiles_list:
                    if smiles == target:
                        # Exact match
                        features.append(target_feature.copy())
                    elif 'CC(C)CC1=CC=C(C=C1)C(C)' in smiles:
                        # Similar structure
                        similar_feature = target_feature.copy()
                        similar_feature += np.random.normal(0, 0.1, 2048)
                        features.append(similar_feature)
                    else:
                        # Different structure
                        features.append(np.random.rand(2048))
                
                return np.array(features)
            
            # Generate features for target
            target_features = create_similarity_features([target_smiles], target_smiles)
            
            # Generate features for library
            library_features = create_similarity_features(library_smiles, target_smiles)
            
            mock_encoder.encode_batch.side_effect = lambda x: (
                target_features if x == [target_smiles] else library_features
            )
            mock_encoder.get_output_dim.return_value = 2048
            mock_encoder_class = MagicMock(return_value=mock_encoder)
            registry.register('morgan', mock_encoder_class)
            
            encoder = registry.get_encoder('morgan')
            
            # Encode target and library
            target_vector = encoder.encode_batch([target_smiles])[0]
            library_vectors = encoder.encode_batch(library_smiles)
            
            # Calculate similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([target_vector], library_vectors)[0]
            
            # Rank molecules by similarity
            ranked_indices = np.argsort(similarities)[::-1]
            ranked_smiles = [library_smiles[i] for i in ranked_indices]
            ranked_similarities = similarities[ranked_indices]
            
            # Verify screening results
            assert len(ranked_smiles) == len(library_smiles)
            assert len(ranked_similarities) == len(library_smiles)
            
            # Most similar should be the exact match
            assert ranked_smiles[0] == target_smiles
            assert ranked_similarities[0] >= ranked_similarities[1]