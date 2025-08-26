"""UniMol encoder implementation with smart environment management.

This module provides a comprehensive UniMol encoder with:
- Intelligent environment detection and configuration
- Process isolation for dependency conflicts
- Cloud API integration as fallback
- Smart dependency management
- Performance optimization
"""

import numpy as np
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
import logging

from molenc.core.base import BaseEncoder
from molenc.core.registry import register_encoder
from molenc.core.exceptions import InvalidSMILESError

# Import smart environment management
try:
    from molenc.isolation import get_environment_manager, IsolationEnvironmentType as EnvironmentType
    SMART_ENVIRONMENT_AVAILABLE = True
except ImportError:
    SMART_ENVIRONMENT_AVAILABLE = False
    get_environment_manager = None
    EnvironmentType = None

# Import advanced dependency management
try:
    from molenc.environments.advanced_dependency_manager import check_encoder_readiness
    ADVANCED_DEPENDENCY_CHECK_AVAILABLE = True
except ImportError:
    ADVANCED_DEPENDENCY_CHECK_AVAILABLE = False

# Import cloud API support
try:
    from molenc.cloud.api_client import get_cloud_client
    CLOUD_API_AVAILABLE = True
except ImportError:
    CLOUD_API_AVAILABLE = False


@register_encoder('unimol')
class UniMolEncoder(BaseEncoder):
    """Advanced UniMol encoder with smart environment management.

    UniMol is a unified molecular representation learning framework that
    combines 2D molecular graphs and 3D molecular conformations to learn
    comprehensive molecular representations.

    This implementation includes:
    - Smart environment detection and configuration
    - Process isolation for dependency conflicts
    - Cloud API fallback for unavailable local installations
    - Intelligent dependency management
    - Performance optimization
    """

    def __init__(self,
                 model_name: str = "unimol_v1",
                 output_dim: int = 512,
                 use_3d: bool = True,
                 max_atoms: int = 512,
                 device: Optional[str] = None,
                 enable_cloud_fallback: bool = True,
                 enable_process_isolation: bool = True,
                 **kwargs) -> None:
        """
        Initialize UniMol encoder with smart environment management.

        Args:
            model_name: Name of the pre-trained UniMol model (default: "unimol_v1")
            output_dim: Output embedding dimension (default: 512)
            use_3d: Whether to use 3D conformations (default: True)
            max_atoms: Maximum number of atoms to process (default: 512)
            device: Device to run the model on ("cpu", "cuda", or None for auto)
            enable_cloud_fallback: Enable cloud API fallback (default: True)
            enable_process_isolation: Enable process isolation (default: True)
            **kwargs: Additional parameters passed to BaseEncoder
        """
        super().__init__(**kwargs)

        self.logger = logging.getLogger(__name__)

        # Store configuration
        self.model_name = model_name
        self.output_dim = output_dim
        self.use_3d = use_3d
        self.max_atoms = max_atoms
        self.enable_cloud_fallback = enable_cloud_fallback
        self.enable_process_isolation = enable_process_isolation

        # Check encoder readiness if advanced dependency check is available
        if ADVANCED_DEPENDENCY_CHECK_AVAILABLE:
            try:
                is_ready, capability_level, status_msg = check_encoder_readiness('unimol')
                self.logger.info(f"UniMol encoder readiness: {status_msg}")
                if not is_ready:
                    self.logger.warning("UniMol dependencies not fully available in current environment")
            except Exception as e:
                self.logger.warning(f"Failed to check UniMol readiness: {e}")

        # Initialize smart environment manager if available
        if SMART_ENVIRONMENT_AVAILABLE:
            self.env_manager = get_environment_manager()
            # Configure environment for UniMol
            self.env_config = self.env_manager.configure_encoder_environment(
                'unimol',
                preferred_environment=EnvironmentType.PROCESS_ISOLATED if enable_process_isolation 
                else EnvironmentType.LOCAL,
                auto_configure=True
            )
        else:
            self.env_manager = None  # type: ignore
            self.env_config = None  # type: ignore

        # Initialize cloud client if enabled
        if enable_cloud_fallback and CLOUD_API_AVAILABLE:
            try:
                self.cloud_client = get_cloud_client()
                self.cloud_available = self.cloud_client.health_check()
            except Exception as e:
                self.logger.warning(f"Cloud client initialization failed: {e}")
                self.cloud_client = None  # type: ignore
                self.cloud_available = False
        else:
            self.cloud_client = None  # type: ignore
            self.cloud_available = False

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self._is_real_model = False
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """
        Load pre-trained UniMol model with smart environment management.

        Uses process isolation or cloud fallback when needed.
        """
        try:
            # Try to load real UniMol model in current environment
            return self._load_real_unimol()
        except ImportError:
            self.logger.warning(
                "unimol_tools not available in current environment.")
            
            # Try process isolation if enabled
            if self.enable_process_isolation and self.env_manager:
                try:
                    self.logger.info("Attempting to load UniMol via process isolation")
                    # This will be handled at encoding time
                    self._is_real_model = False
                    return self._load_placeholder_model()
                except Exception as e:
                    self.logger.warning(f"Process isolation failed: {e}")
            
            # Try cloud API if enabled
            if self.cloud_available:
                try:
                    self.logger.info("Using cloud-based UniMol encoder as fallback")
                    self._is_real_model = True
                    # Model will be loaded on demand
                    return None  # type: ignore
                except Exception as e:
                    self.logger.warning(f"Cloud encoder setup failed: {e}")
            
            # Fall back to placeholder
            self.logger.info("Using placeholder UniMol implementation.")
            self.logger.info("Install with: pip install unimol_tools")
            return self._load_placeholder_model()
            
        except Exception as e:
            self.logger.warning(
                f"Failed to load real UniMol model ({e}), using placeholder.")
            return self._load_placeholder_model()

    def _load_real_unimol(self) -> Any:
        """
        Load real UniMol model using unimol_tools.
        """
        try:
            from unimol_tools import UniMolRepr

            # Initialize UniMolRepr
            model = UniMolRepr(
                data_type='molecule',
                remove_hs=(not self.use_3d),  # Keep hydrogens for 3D mode
                use_gpu=(self.device.type == "cuda")
            )

            print(f"✓ Successfully loaded real Uni-Mol model (device: {self.device})")
            self._is_real_model = True
            return model
        except ImportError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to initialize UniMol model: {e}")

    def _load_placeholder_model(self) -> nn.Module:
        """
        Load placeholder model for demonstration.
        """
        print(f"Warning: Using placeholder UniMol implementation. "
              f"Please integrate with actual UniMol model for production use.")

        # Simple placeholder model
        model = self._create_placeholder_model()
        model.to(self.device)
        model.eval()
        self._is_real_model = False
        return model

    def _create_placeholder_model(self) -> nn.Module:
        """
        Create a placeholder model for demonstration.

        Returns:
            Placeholder PyTorch model
        """
        class PlaceholderUniMol(nn.Module):
            def __init__(self, output_dim: int) -> None:
                super().__init__()
                # Simple MLP as placeholder
                self.atom_embedding = nn.Embedding(100, 64)  # For atomic numbers
                self.mlp = nn.Sequential(
                    nn.Linear(64, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, output_dim)
                )

            def forward(self, atom_types: torch.Tensor, coords_3d: Optional[torch.Tensor] = None) -> torch.Tensor:
                # Simple atom type embedding
                atom_embeds = self.atom_embedding(atom_types)  # Shape: [num_atoms, 64]

                # If 3D coordinates are provided, incorporate them
                if coords_3d is not None:
                    # Simple distance-based features (placeholder)
                    # Use mean distance to all other atoms as a simple feature
                    if coords_3d.size(0) > 1:
                        # [num_atoms, num_atoms]
                        distances = torch.cdist(coords_3d, coords_3d)
                        # Get mean distance for each atom (excluding self-distance)
                        mask = torch.eye(distances.size(
                            0), device=distances.device).bool()
                        distances.masked_fill_(mask, float('inf'))
                        mean_distances = distances.mean(
                            dim=-1, keepdim=True)  # [num_atoms, 1]
                        mean_distances = torch.where(torch.isinf(
                            mean_distances), torch.zeros_like(mean_distances), mean_distances)
                    else:
                        mean_distances = torch.zeros(
                            coords_3d.size(0), 1, device=coords_3d.device)

                    # Expand distance features to match embedding dimension
                    dist_features = mean_distances.expand(-1, 64)  # [num_atoms, 64]
                    atom_embeds = atom_embeds + 0.1 * dist_features  # Simple additive combination

                # Pool atom embeddings (mean pooling)
                pooled = atom_embeds.mean(dim=0)  # Shape: [64]

                # Apply MLP
                output = self.mlp(pooled)  # Shape: [output_dim]

                return output

        return PlaceholderUniMol(self.output_dim)

    def _mol_to_features(self, mol) -> Dict[str, torch.Tensor]:
        """
        Convert RDKit molecule to features for UniMol.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary containing molecular features
        """
        # Get atom types
        atom_types: List[int] = []
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetAtomicNum())

        atom_types_tensor = torch.tensor(atom_types, dtype=torch.long)

        features: Dict[str, torch.Tensor] = {'atom_types': atom_types_tensor}

        # Get 3D coordinates if requested
        if self.use_3d:
            coords_3d = self._get_3d_coordinates(mol)
            if coords_3d is not None:
                features['coords_3d'] = coords_3d

        return features

    def _get_3d_coordinates(self, mol) -> Optional[torch.Tensor]:
        """
        Generate or extract 3D coordinates for the molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            3D coordinates tensor or None if generation fails
        """
        try:
            # Create a copy to avoid modifying the original
            mol_copy = Chem.Mol(mol)

            # Add hydrogens for better 3D generation
            mol_with_h = Chem.AddHs(mol_copy)

            # Generate 3D conformation
            result = AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
            if result != 0:
                # Fallback: try without random seed
                result = AllChem.EmbedMolecule(mol_with_h)
                if result != 0:
                    return None

            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol_with_h)

            # Extract coordinates only for heavy atoms (original molecule atoms)
            conf = mol_with_h.GetConformer()
            coords: List[List[float]] = []

            # Map heavy atoms from mol_with_h back to original mol
            heavy_atom_idx = 0
            for i in range(mol_with_h.GetNumAtoms()):
                atom = mol_with_h.GetAtomWithIdx(i)
                if atom.GetAtomicNum() != 1:  # Skip hydrogens
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                    heavy_atom_idx += 1

                    # Stop when we have all heavy atoms
                    if heavy_atom_idx >= mol.GetNumAtoms():
                        break

            return torch.tensor(coords, dtype=torch.float32)

        except Exception:
            return None

    def _encode_single(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES string using UniMol with intelligent routing.

        Args:
            smiles: SMILES string to encode

        Returns:
            UniMol embedding as numpy array

        Raises:
            InvalidSMILESError: If SMILES is invalid
        """
        try:
            # Check for empty SMILES
            if not smiles or not smiles.strip():
                raise InvalidSMILESError(smiles, "Empty SMILES string")

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise InvalidSMILESError(smiles, "Could not parse SMILES")

            # Check atom count
            if mol.GetNumAtoms() > self.max_atoms:
                raise InvalidSMILESError(
                    smiles,
                    f"Molecule has {mol.GetNumAtoms()} atoms, exceeds maximum of {self.max_atoms}"
                )

            # Try to encode with real model first
            if self._is_real_model and self.model is not None:
                return self._encode_with_real_model(smiles)
            
            # Try process isolation if enabled
            elif self.enable_process_isolation and self.env_manager:
                try:
                    return self._encode_with_process_isolation(smiles)
                except Exception as e:
                    self.logger.warning(f"Process isolation failed for {smiles}: {e}")
            
            # Try cloud API if available
            elif self.cloud_available and self.cloud_client:
                try:
                    return self._encode_with_cloud_api(smiles)
                except Exception as e:
                    self.logger.warning(f"Cloud API failed for {smiles}: {e}")
            
            # Fall back to placeholder model
            return self._encode_with_placeholder_model(smiles, mol)

        except Exception as e:
            if isinstance(e, InvalidSMILESError):
                raise e
            raise InvalidSMILESError(smiles, f"UniMol encoding failed: {str(e)}")

    def _encode_with_real_model(self, smiles: str) -> np.ndarray:
        """
        Encode using real UniMol model.
        """
        try:
            # Use real UniMol model
            result = self.model.get_repr([smiles])
            
            # Debug: Print the type and keys of result
            self.logger.debug(f"Result type: {type(result)}")
            if hasattr(result, 'keys'):
                self.logger.debug(f"Result keys: {list(result.keys())}")
            
            # Handle different output formats from unimol_tools
            if isinstance(result, dict):
                # Newer versions return a dict
                if 'cls_repr' in result:
                    # Extract CLS token representation
                    embedding = np.array(result['cls_repr'][0])
                elif 'representations' in result:
                    # Alternative key for representations
                    embedding = np.array(result['representations'][0])
                else:
                    # If we can't find the right key, use the first one
                    first_key = list(result.keys())[0]
                    embedding = np.array(result[first_key][0])
            else:
                # Older versions might return a list or other structure
                # Assume the first element is what we want
                embedding = np.array(result[0])
            
            # Ensure embedding is the right type
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Adjust output dimension if needed
            if len(embedding) != self.output_dim:
                embedding = self._adjust_dimension(embedding)

            return embedding.astype(np.float32)

        except Exception as e:
            self.logger.error(f"Real UniMol encoding failed: {e}")
            raise RuntimeError(f"Real UniMol encoding failed: {e}")

    def _encode_with_process_isolation(self, smiles: str) -> np.ndarray:
        """
        Encode using process isolation.
        """
        if not self.env_manager:
            raise RuntimeError("Environment manager not available")
        
        try:
            result = self.env_manager.execute_encoder(
                'unimol',
                smiles,
                {
                    'model_name': self.model_name,
                    'output_dim': self.output_dim,
                    'use_3d': self.use_3d,
                    'max_atoms': self.max_atoms
                }
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Process isolation encoding failed: {e}")

    def _encode_with_cloud_api(self, smiles: str) -> np.ndarray:
        """
        Encode using cloud API.
        """
        try:
            response = self.cloud_client.encode_single(
                smiles=smiles,
                encoder_type='unimol',
                options={
                    'model_name': self.model_name,
                    'output_dim': self.output_dim,
                    'use_3d': self.use_3d
                }
            )

            if not response.success:
                raise RuntimeError(f"Cloud API error: {response.error_message}")

            embedding = np.array(response.embeddings, dtype=np.float32)

            # Adjust dimension if needed
            if len(embedding) != self.output_dim:
                embedding = self._adjust_dimension(embedding)

            return embedding

        except Exception as e:
            raise RuntimeError(f"Cloud API encoding failed: {e}")

    def _encode_with_placeholder_model(self, smiles: str, mol) -> np.ndarray:
        """
        Encode using placeholder model.
        """
        try:
            # Convert molecule to features
            features = self._mol_to_features(mol)

            # Move features to device
            for key, value in features.items():
                features[key] = value.to(self.device)

            # Get embeddings
            with torch.no_grad():
                embedding = self.model(
                    features['atom_types'],
                    features.get('coords_3d', None)
                )

            # Convert to numpy
            embedding = embedding.cpu().numpy()

            return embedding.astype(np.float32)

        except Exception as e:
            raise RuntimeError(f"Placeholder UniMol encoding failed: {e}")

    def _adjust_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """
        Adjust embedding dimension to match output_dim.
        """
        current_dim = len(embedding)

        if current_dim > self.output_dim:
            # Truncate to target dimension
            return embedding[:self.output_dim]
        elif current_dim < self.output_dim:
            # Zero-pad to target dimension
            padded = np.zeros(self.output_dim)
            padded[:current_dim] = embedding
            return padded
        else:
            return embedding

    def encode_batch(self, smiles_list: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of SMILES strings with intelligent processing.

        Args:
            smiles_list: List of SMILES strings to encode
            batch_size: Batch size for processing

        Returns:
            Numpy array of embeddings with shape (n_molecules, output_dim)
        """
        if not smiles_list:
            return np.empty((0, self.output_dim), dtype=np.float32)

        # Try real model first
        if self._is_real_model and self.model is not None:
            try:
                return self._encode_batch_with_real_model(smiles_list)
            except Exception as e:
                self.logger.warning(f"Real model batch encoding failed: {e}")
        
        # Try process isolation
        elif self.enable_process_isolation and self.env_manager:
            try:
                return self._encode_batch_with_process_isolation(smiles_list)
            except Exception as e:
                self.logger.warning(f"Process isolation batch encoding failed: {e}")
        
        # Try cloud API
        elif self.cloud_available and self.cloud_client:
            try:
                return self._encode_batch_with_cloud_api(smiles_list)
            except Exception as e:
                self.logger.warning(f"Cloud API batch encoding failed: {e}")
        
        # Fall back to placeholder model
        return self._encode_batch_with_placeholder_model(smiles_list)

    def _encode_batch_with_real_model(self, smiles_list: List[str]) -> np.ndarray:
        """
        Batch encode using real UniMol model.
        """
        try:
            # Filter valid SMILES
            valid_smiles = []
            valid_indices = []

            for i, smiles in enumerate(smiles_list):
                try:
                    if not smiles or not smiles.strip():
                        continue
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None or mol.GetNumAtoms() > self.max_atoms:
                        continue
                    valid_smiles.append(smiles)
                    valid_indices.append(i)
                except Exception:
                    continue

            if not valid_smiles:
                return np.zeros((len(smiles_list), self.output_dim), dtype=np.float32)

            # Batch encode with real model
            result = self.model.get_repr(valid_smiles)
            
            # Handle different output formats from unimol_tools
            if isinstance(result, dict):
                # Newer versions return a dict
                if 'cls_repr' in result:
                    # Extract CLS token representation
                    embeddings = np.array(result['cls_repr'])
                elif 'representations' in result:
                    # Alternative key for representations
                    embeddings = np.array(result['representations'])
                else:
                    # If we can't find the right key, use the first one
                    first_key = list(result.keys())[0]
                    embeddings = np.array(result[first_key])
            else:
                # Older versions might return a list or other structure
                # Assume the first element is what we want
                embeddings = np.array(result)
            
            # Ensure embeddings is the right type
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Adjust dimensions if needed
            if embeddings.ndim == 1:
                # If we got a 1D array, reshape it
                embeddings = embeddings.reshape(1, -1)
            
            if embeddings.ndim > 2:
                # If we got a multi-dimensional array, flatten it to 2D
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            
            if embeddings.shape[1] != self.output_dim:
                adjusted_embeddings = np.zeros((len(embeddings), self.output_dim))
                for i, emb in enumerate(embeddings):
                    adjusted_embeddings[i] = self._adjust_dimension(emb)
                embeddings = adjusted_embeddings

            # Create full result array
            full_embeddings = np.zeros(
                (len(smiles_list), self.output_dim), dtype=np.float32)
            for i, valid_idx in enumerate(valid_indices):
                full_embeddings[valid_idx] = embeddings[i]

            return full_embeddings

        except Exception as e:
            raise RuntimeError(f"Real UniMol batch encoding failed: {e}")

    def _encode_batch_with_process_isolation(self, smiles_list: List[str]) -> np.ndarray:
        """
        Batch encode using process isolation.
        """
        if not self.env_manager:
            raise RuntimeError("Environment manager not available")
        
        try:
            result = self.env_manager.execute_encoder(
                'unimol',
                smiles_list,
                {
                    'model_name': self.model_name,
                    'output_dim': self.output_dim,
                    'use_3d': self.use_3d,
                    'max_atoms': self.max_atoms
                }
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Process isolation batch encoding failed: {e}")

    def _encode_batch_with_cloud_api(self, smiles_list: List[str]) -> np.ndarray:
        """
        Batch encode using cloud API.
        """
        try:
            response = self.cloud_client.encode_batch(
                smiles_list=smiles_list,
                encoder_type='unimol',
                options={
                    'model_name': self.model_name,
                    'output_dim': self.output_dim,
                    'use_3d': self.use_3d
                }
            )

            if not response.success:
                raise RuntimeError(f"Cloud API error: {response.error_message}")

            embeddings = np.array(response.embeddings, dtype=np.float32)

            # Adjust dimensions if needed
            if embeddings.shape[1] != self.output_dim:
                adjusted_embeddings = []
                for embedding in embeddings:
                    adjusted_embeddings.append(self._adjust_dimension(embedding))
                embeddings = np.array(adjusted_embeddings, dtype=np.float32)

            return embeddings

        except Exception as e:
            raise RuntimeError(f"Cloud API batch encoding failed: {e}")

    def _encode_batch_with_placeholder_model(self, smiles_list: List[str]) -> np.ndarray:
        """
        Batch encode using placeholder model.
        """
        embeddings = []

        for smiles in smiles_list:
            try:
                embedding = self._encode_single(smiles)
                embeddings.append(embedding)
            except Exception:
                # Add zero vector for failed encodings
                embeddings.append(np.zeros(self.output_dim, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)

    def get_output_dim(self) -> int:
        """
        Get the output dimension of UniMol embeddings.

        Returns:
            Output dimension
        """
        return self.output_dim

    def get_config(self) -> Dict[str, Any]:
        """
        Get encoder configuration.

        Returns:
            Configuration dictionary
        """
        config: Dict[str, Any] = super().get_config()
        config.update({
            'encoder_type': 'unimol',
            'model_name': self.model_name,
            'output_dim': self.output_dim,
            'use_3d': self.use_3d,
            'max_atoms': self.max_atoms,
            'device': str(self.device),
            'is_real_model': self._is_real_model,
            'cloud_fallback_enabled': self.enable_cloud_fallback,
            'process_isolation_enabled': self.enable_process_isolation,
            'cloud_available': self.cloud_available
        })

        return config

    def get_status(self) -> Dict[str, Any]:
        """
        Get current encoder status and capabilities.

        Returns:
            Dictionary containing status information
        """
        status: Dict[str, Any] = {
            'initialized': hasattr(self, 'model') and self.model is not None,
            'model_type': 'real' if self._is_real_model else 'placeholder',
            'device': str(self.device),
            'output_dim': self.output_dim,
            'cloud_fallback_enabled': self.enable_cloud_fallback,
            'process_isolation_enabled': self.enable_process_isolation,
            'cloud_available': self.cloud_available
        }

        if self.env_manager and self.env_config:
            status['environment_type'] = self.env_config.environment_type.value
            status['capability_level'] = self.env_config.capability_level.value

        return status

    def get_feature_names(self) -> List[str]:
        """
        Get feature names for the embedding dimensions.

        Returns:
            List of feature names
        """
        return [f"unimol_dim_{i}" for i in range(self.output_dim)]

    def __repr__(self) -> str:
        return (
            f"UniMolEncoder(model_name='{self.model_name}', "
            f"output_dim={self.output_dim}, use_3d={self.use_3d})"
        )
