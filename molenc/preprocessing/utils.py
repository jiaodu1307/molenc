"""Preprocessing utility functions."""

import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from collections import defaultdict

from .standardize import SMILESStandardizer
from .filters import MolecularFilters
from .validators import SMILESValidator
from molenc.core.exceptions import InvalidSMILESError, DependencyError


def preprocess_smiles_list(smiles_list: List[str],
                           standardize: bool = True,
                           validate: bool = True,
                           filter_molecules: bool = True,
                           standardize_options: Optional[Dict] = None,
                           validation_options: Optional[Dict] = None,
                           filter_options: Optional[Dict] = None,
                           standardizer_config: Optional[Dict] = None,
                           validator_config: Optional[Dict] = None,
                           filter_config: Optional[Dict] = None,
                           parallel: bool = False,
                           n_jobs: int = 1,
                           return_details: bool = False) -> Dict[str, Any]:
    """
    Comprehensive preprocessing pipeline for SMILES lists.

    Args:
        smiles_list: List of SMILES strings to preprocess
        standardize: Whether to standardize SMILES
        validate: Whether to validate SMILES
        filter_molecules: Whether to filter molecules
        standardize_options: Configuration for SMILESStandardizer (preferred)
        validation_options: Configuration for SMILESValidator (preferred)
        filter_options: Configuration for MolecularFilters (preferred)
        standardizer_config: Configuration for SMILESStandardizer (deprecated)
        validator_config: Configuration for SMILESValidator (deprecated)
        filter_config: Configuration for MolecularFilters (deprecated)
        parallel: Whether to use parallel processing
        n_jobs: Number of parallel jobs (1 for sequential processing)
        return_details: Whether to return detailed processing information

    Returns:
        Dictionary with processing results
    """
    if not smiles_list:
        return {
            'processed_smiles': [],
            'failed_indices': [],
            'stats': {'total': 0, 'processed': 0, 'failed': 0}
        }

    # Handle parameter compatibility
    std_config = standardize_options or standardizer_config or {}
    val_config = validation_options or validator_config or {}
    filt_config = filter_options or filter_config or {}

    # Use parallel processing if specified
    if parallel and n_jobs == 1:
        n_jobs = mp.cpu_count()

    # Initialize components with dependency checking
    standardizer = None
    validator = None
    mol_filter = None

    if standardize:
        try:
            standardizer = SMILESStandardizer(**std_config)
        except DependencyError:
            logging.warning("RDKit not available, skipping standardization")
            standardize = False

    if validate:
        try:
            validator = SMILESValidator(**val_config)
        except DependencyError:
            logging.warning("RDKit not available, skipping validation")
            validate = False

    if filter_molecules:
        try:
            mol_filter = MolecularFilters(**filt_config)
        except DependencyError:
            logging.warning("RDKit not available, skipping filtering")
            filter_molecules = False

    # Processing summary
    summary = {
        'total': len(smiles_list),
        'processed': 0,
        'failed': 0,
        'standardization': {'enabled': standardize, 'processed': 0, 'failed': 0},
        'validation': {'enabled': validate, 'processed': 0, 'failed': 0},
        'filtering': {'enabled': filter_molecules, 'processed': 0, 'failed': 0},
        'rejection_reasons': {
            'standardization_failed': 0,
            'validation_failed': 0,
            'filtering_failed': 0,
            'processing_error': 0
        },
        'details': [] if return_details else None
    }

    if n_jobs == 1:
        # Sequential processing
        processed_smiles = []
        failed_indices = []

        for i, smiles in enumerate(smiles_list):
            result = _process_single_smiles(
                smiles, i, standardizer, validator, mol_filter, return_details
            )

            if result['success']:
                processed_smiles.append(result['smiles'])
                summary['processed'] += 1
            else:
                failed_indices.append(i)
                summary['failed'] += 1
                reason = result.get('reason', 'unknown')
                if reason in summary['rejection_reasons']:
                    summary['rejection_reasons'][reason] += 1
                else:
                    summary['rejection_reasons']['processing_error'] += 1

            if return_details:
                summary['details'].append(result)

    else:
        # Parallel processing
        processed_smiles = []
        failed_indices = []

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    _process_single_smiles,
                    smiles, i, standardizer, validator, mol_filter, return_details
                ): i for i, smiles in enumerate(smiles_list)
            }

            # Collect results
            results = [None] * len(smiles_list)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = {
                        'success': False,
                        'reason': 'processing_error',
                        'error': str(e),
                        'original_index': index
                    }

            # Process results in order
            for i, result in enumerate(results):
                if result['success']:
                    processed_smiles.append(result['smiles'])
                    summary['processed'] += 1
                else:
                    failed_indices.append(i)
                    summary['failed'] += 1
                    reason = result.get('reason', 'unknown')
                    if reason in summary['rejection_reasons']:
                        summary['rejection_reasons'][reason] += 1
                    else:
                        summary['rejection_reasons']['processing_error'] += 1

                if return_details:
                    summary['details'].append(result)

    # Calculate rates
    summary['success_rate'] = summary['processed'] / \
        summary['total'] if summary['total'] > 0 else 0
    summary['rejection_rate'] = summary['failed'] / \
        summary['total'] if summary['total'] > 0 else 0

    return {
        'processed_smiles': processed_smiles,
        'failed_indices': failed_indices,
        'stats': summary
    }


def _process_single_smiles(smiles: str,
                           index: int,
                           standardizer: Optional[SMILESStandardizer],
                           validator: Optional[SMILESValidator],
                           mol_filter: Optional[MolecularFilters],
                           return_details: bool) -> Dict[str, Any]:
    """
    Process a single SMILES string through the preprocessing pipeline.

    Args:
        smiles: SMILES string to process
        index: Original index in the input list
        standardizer: SMILESStandardizer instance
        validator: SMILESValidator instance
        mol_filter: MolecularFilters instance
        return_details: Whether to return detailed information

    Returns:
        Dictionary with processing results
    """
    result = {
        'success': False,
        'smiles': smiles,
        'original_index': index,
        'reason': None,
        'error': None
    }

    try:
        current_smiles = smiles

        # Step 1: Standardization
        if standardizer:
            try:
                current_smiles = standardizer.standardize(current_smiles)
                if return_details:
                    result['standardized_smiles'] = current_smiles
            except InvalidSMILESError as e:
                result['reason'] = 'standardization_failed'
                result['error'] = str(e)
                return result

        # Step 2: Validation
        if validator:
            is_valid, validation_results = validator.validate(current_smiles)
            if not is_valid:
                result['reason'] = 'validation_failed'
                result['error'] = validation_results.get('errors', [])
                if return_details:
                    result['validation_results'] = validation_results
                return result
            elif return_details:
                result['validation_results'] = validation_results

        # Step 3: Filtering
        if mol_filter:
            passes_filter, filter_results = mol_filter.filter_molecule(current_smiles)
            if not passes_filter:
                result['reason'] = 'filtering_failed'
                result['error'] = filter_results
                if return_details:
                    result['filter_results'] = filter_results
                return result
            elif return_details:
                result['filter_results'] = filter_results

        # Success
        result['success'] = True
        result['smiles'] = current_smiles

    except Exception as e:
        result['reason'] = 'processing_error'
        result['error'] = str(e)

    return result


def batch_standardize(smiles_list: List[str],
                      n_jobs: int = 1,
                      **standardizer_kwargs) -> List[str]:
    """
    Batch standardization of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        n_jobs: Number of parallel jobs
        **standardizer_kwargs: Arguments for SMILESStandardizer

    Returns:
        List of standardized SMILES strings
    """
    try:
        standardizer = SMILESStandardizer(**standardizer_kwargs)
    except DependencyError:
        logging.warning("RDKit not available, returning original SMILES")
        return smiles_list

    if n_jobs == 1:
        standardized_smiles, _ = standardizer.standardize_batch(smiles_list)
        return standardized_smiles

    # Parallel processing
    chunk_size = max(1, len(smiles_list) // n_jobs)
    chunks = [smiles_list[i:i + chunk_size]
              for i in range(0, len(smiles_list), chunk_size)]

    standardized_smiles = []

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = []

        for chunk in chunks:
            future = executor.submit(standardizer.standardize_batch, chunk)
            futures.append(future)

        for future in futures:
            chunk_standardized, _ = future.result()
            standardized_smiles.extend(chunk_standardized)

    return standardized_smiles


def create_preprocessing_pipeline(standardize: bool = False,
                                  validate: bool = False,
                                  filter_molecules: bool = False,
                                  standardize_options: Optional[Dict[str, Any]] = None,
                                  validation_options: Optional[Dict[str, Any]] = None,
                                  filter_options: Optional[Dict[str, Any]] = None,
                                  steps: Optional[List[str]] = None,
                                  configs: Optional[Dict[str, Dict]] = None) -> Callable:
    """
    Create a custom preprocessing pipeline.

    Args:
        standardize: Whether to standardize SMILES
        validate: Whether to validate SMILES
        filter_molecules: Whether to filter molecules
        standardize_options: Configuration for SMILESStandardizer
        validation_options: Configuration for SMILESValidator
        filter_options: Configuration for MolecularFilters
        steps: List of preprocessing steps (legacy parameter)
        configs: Configuration dictionaries for each step (legacy parameter)

    Returns:
        Preprocessing function
    """
    # Handle legacy parameters
    if steps is not None:
        standardize = 'standardize' in steps
        validate = 'validate' in steps
        filter_molecules = 'filter' in steps

    if configs is not None:
        standardize_options = standardize_options or configs.get('standardize')
        validation_options = validation_options or configs.get('validate')
        filter_options = filter_options or configs.get('filter')

    def pipeline(smiles: str) -> Dict[str, Any]:
        """
        Apply the preprocessing pipeline to a single SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with processing results
        """
        return _process_single_smiles_legacy(
            smiles,
            standardize=standardize,
            validate=validate,
            filter_molecules=filter_molecules,
            standardize_options=standardize_options,
            validation_options=validation_options,
            filter_options=filter_options
        )

    return pipeline


def get_preprocessing_stats(smiles_list: List[str],
                            standardizer_config: Optional[Dict] = None,
                            validator_config: Optional[Dict] = None,
                            filter_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get preprocessing statistics without actually processing the data.

    Args:
        smiles_list: List of SMILES strings
        standardizer_config: Configuration for SMILESStandardizer
        validator_config: Configuration for SMILESValidator
        filter_config: Configuration for MolecularFilters

    Returns:
        Dictionary with preprocessing statistics
    """
    stats = {
        'total': len(smiles_list),
        'processed': 0,
        'failed': 0,
        'rejection_rate': 0.0
    }

    # Validation stats
    if validator_config is not None:
        validator = SMILESValidator(**validator_config)
        valid_smiles, invalid_indices, _ = validator.validate_batch(smiles_list)
        stats['validation'] = validator.get_stats()
        stats['valid_after_validation'] = len(valid_smiles)
        stats['failed'] += len(invalid_indices)

    # Filter stats
    if filter_config is not None:
        mol_filter = MolecularFilters(**filter_config)
        filter_stats = mol_filter.get_filter_stats(smiles_list)
        stats['filtering'] = filter_stats

    # Calculate processed and rejection rate
    stats['processed'] = stats['total'] - stats['failed']
    if stats['total'] > 0:
        stats['rejection_rate'] = stats['failed'] / stats['total']

    return stats


def save_preprocessing_report(smiles_list: List[str],
                              processed_smiles: List[str],
                              summary: Dict[str, Any],
                              output_path: str):
    """
    Save a preprocessing report to file.

    Args:
        smiles_list: Original SMILES list
        processed_smiles: Processed SMILES list
        summary: Processing summary
        output_path: Path to save the report
    """
    import json
    from datetime import datetime

    report = {
        'timestamp': datetime.now().isoformat(),
        'input_count': len(smiles_list),
        'output_count': len(processed_smiles),
        'summary': summary,
        'sample_input': smiles_list[:10] if smiles_list else [],
        'sample_output': processed_smiles[:10] if processed_smiles else []
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logging.info(f"Preprocessing report saved to {output_path}")


def _process_single_smiles_legacy(smiles: str,
                                  index: int = 0,
                                  standardize: bool = False,
                                  validate: bool = False,
                                  filter_molecules: bool = False,
                                  standardize_options: Optional[Dict[str, Any]] = None,
                                  validation_options: Optional[Dict[str, Any]] = None,
                                  filter_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Legacy wrapper for _process_single_smiles with old parameter format.

    This function provides backward compatibility for tests that use the old
    parameter format.
    """
    # Initialize components based on flags
    standardizer = None
    validator = None
    mol_filter = None

    try:
        if standardize:
            from molenc.preprocessing.standardize import SMILESStandardizer
            standardizer = SMILESStandardizer(**(standardize_options or {}))
    except DependencyError:
        pass

    try:
        if validate:
            from molenc.preprocessing.validators import SMILESValidator
            validator = SMILESValidator(**(validation_options or {}))
    except DependencyError:
        pass

    try:
        if filter_molecules:
            from molenc.preprocessing.filters import MolecularFilters
            mol_filter = MolecularFilters(**(filter_options or {}))
    except DependencyError:
        pass

    # Call the actual function
    result = _process_single_smiles_original(
        smiles=smiles,
        index=index,
        standardizer=standardizer,
        validator=validator,
        mol_filter=mol_filter,
        return_details=True
    )

    # Convert result format for backward compatibility
    return {
        'success': result['success'],
        'smiles': result.get('smiles'),  # Keep 'smiles' key for consistency
        'processed_smiles': result.get('smiles') if result['success'] else None,
        'error_stage': result.get('reason'),
        'error_message': result.get('error'),
        'reason': result.get('reason'),  # Add 'reason' key for consistency
        'index': result.get('index', index)  # Add 'index' key for consistency
    }


# Create alias for backward compatibility
_process_single_smiles_original = _process_single_smiles
_process_single_smiles = _process_single_smiles_legacy
