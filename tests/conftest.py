"""Shared test configuration and fixtures for MolEnc tests."""

import pytest
import numpy as np
from typing import List, Dict, Any
import tempfile
import os
from unittest.mock import Mock, patch

# Test data
SAMPLE_SMILES = [
    "CCO",  # ethanol
    "CC(=O)O",  # acetic acid
    "c1ccccc1",  # benzene
    "CCN(CC)CC",  # triethylamine
    "CC(C)C",  # isobutane
    "C1CCCCC1",  # cyclohexane
    "CC(C)(C)O",  # tert-butanol
    "CCCCCCCCCCCCCCCCCC(=O)O",  # stearic acid
]

INVALID_SMILES = [
    "invalid_smiles",
    "C(C(C",  # unmatched parentheses
    "C1CC",  # incomplete ring
    "",  # empty string
    "C#C#C",  # invalid triple bonds
]

SAMPLE_MOLECULES_DATA = {
    "CCO": {
        "name": "ethanol",
        "molecular_weight": 46.07,
        "logp": -0.31,
        "hbd": 1,
        "hba": 1,
    },
    "CC(=O)O": {
        "name": "acetic_acid",
        "molecular_weight": 60.05,
        "logp": -0.17,
        "hbd": 1,
        "hba": 2,
    },
    "c1ccccc1": {
        "name": "benzene",
        "molecular_weight": 78.11,
        "logp": 2.13,
        "hbd": 0,
        "hba": 0,
    },
}


@pytest.fixture
def sample_smiles() -> List[str]:
    """Provide sample SMILES strings for testing."""
    return SAMPLE_SMILES.copy()


@pytest.fixture
def invalid_smiles() -> List[str]:
    """Provide invalid SMILES strings for testing."""
    return INVALID_SMILES.copy()


@pytest.fixture
def sample_molecules_data() -> Dict[str, Dict[str, Any]]:
    """Provide sample molecule data for testing."""
    return SAMPLE_MOLECULES_DATA.copy()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_encoder():
    """Create a mock encoder for testing."""
    encoder = Mock()
    encoder.encode.return_value = np.random.rand(1024)
    encoder.encode_batch.return_value = np.random.rand(5, 1024)
    encoder.get_output_dim.return_value = 1024
    return encoder


@pytest.fixture
def sample_vectors():
    """Provide sample encoding vectors for testing."""
    return {
        "single": np.random.rand(1024),
        "batch": np.random.rand(5, 1024),
        "small": np.random.rand(128),
    }


@pytest.fixture(scope="session")
def rdkit_available():
    """Check if RDKit is available."""
    try:
        import rdkit
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def transformers_available():
    """Check if transformers is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False


@pytest.fixture
def mock_rdkit_mol():
    """Create a mock RDKit molecule object."""
    mol = Mock()
    mol.GetNumAtoms.return_value = 10
    mol.GetNumBonds.return_value = 9
    return mol


@pytest.fixture
def config_dict():
    """Provide sample configuration dictionary."""
    return {
        "encoder_name": "morgan",
        "radius": 2,
        "n_bits": 1024,
        "handle_errors": "raise",
    }


# Skip markers for optional dependencies
skip_if_no_rdkit = pytest.mark.skipif(
    "rdkit" not in globals(),
    reason="RDKit not available"
)

skip_if_no_torch = pytest.mark.skipif(
    "torch" not in globals(),
    reason="PyTorch not available"
)

skip_if_no_transformers = pytest.mark.skipif(
    "transformers" not in globals(),
    reason="transformers not available"
)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["integration", "benchmark", "slow"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add slow marker to tests that might be slow
        if "benchmark" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)