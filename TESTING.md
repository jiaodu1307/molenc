# Testing Guide for molenc

This document provides comprehensive information about the testing framework for the molenc library.

## Overview

The molenc testing framework includes:
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and workflows
- **Benchmark Tests**: Performance and scalability testing
- **Coverage Reports**: Code coverage analysis
- **CI/CD Integration**: Automated testing on GitHub Actions

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── core/                       # Core module tests
│   ├── test_base.py
│   ├── test_registry.py
│   └── test_exceptions.py
├── encoders/                   # Encoder tests
│   ├── test_fingerprints.py
│   ├── test_substructure.py
│   └── test_representations.py
├── preprocessing/              # Preprocessing tests
│   ├── test_standardize.py
│   ├── test_validators.py
│   ├── test_filters.py
│   └── test_utils.py
├── environments/               # Environment management tests
│   ├── test_dependencies.py
│   ├── test_config.py
│   └── test_manager.py
├── integration/                # Integration tests
│   ├── test_encoder_workflows.py
│   └── test_system_integration.py
└── benchmarks/                 # Performance benchmarks
    └── test_performance.py
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=molenc --cov-report=html

# Run specific test types
pytest tests/core/          # Unit tests for core modules
pytest tests/integration/   # Integration tests
pytest tests/benchmarks/    # Benchmark tests
```

### Using pytest

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock pytest-benchmark

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=molenc --cov-report=html

# Run specific test file
pytest tests/core/test_base.py

# Run tests with specific marker
pytest tests/ -m "not slow"
```

### Test Markers

Tests are organized using pytest markers:

- `unit`: Unit tests (default)
- `integration`: Integration tests
- `benchmark`: Performance benchmarks
- `slow`: Tests that take longer to run
- `chemistry`: Tests requiring chemistry dependencies (RDKit)
- `deep_learning`: Tests requiring ML dependencies (PyTorch, etc.)
- `visualization`: Tests requiring plotting dependencies
- `gpu`: Tests requiring GPU
- `network`: Tests requiring network access

```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Run chemistry-related tests
pytest tests/ -m "chemistry"

# Run tests excluding benchmarks
pytest tests/ -m "not benchmark"
```

## Test Configuration

### pytest.ini

The `pytest.ini` file contains default configuration:

```ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    --strict-markers
    --cov=molenc
    --cov-report=term-missing
    --cov-fail-under=80
    --tb=short

testpaths = tests
markers = 
    slow: marks tests as slow
    integration: marks tests as integration tests
    # ... other markers
```

### Coverage Configuration

The `.coveragerc` file configures coverage reporting:

```ini
[run]
source = molenc
omit = 
    */tests/*
    */examples/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

## Writing Tests

### Unit Test Example

```python
import pytest
from unittest.mock import patch, MagicMock
from molenc.core.base import BaseEncoder

class TestBaseEncoder:
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = BaseEncoder()
        assert encoder is not None
    
    @patch('molenc.core.base.some_dependency')
    def test_with_mock(self, mock_dep):
        """Test with mocked dependency."""
        mock_dep.return_value = "mocked_result"
        encoder = BaseEncoder()
        result = encoder.some_method()
        assert result == "expected_result"
```

### Integration Test Example

```python
import pytest
from molenc.preprocessing import preprocess_smiles_list
from molenc.core.registry import EncoderRegistry

@pytest.mark.integration
class TestWorkflow:
    def test_complete_workflow(self):
        """Test complete preprocessing to encoding workflow."""
        smiles = ['CCO', 'CC(=O)O', 'c1ccccc1']
        
        # Preprocess
        processed = preprocess_smiles_list(smiles)
        
        # Encode
        registry = EncoderRegistry()
        encoder = registry.get_encoder('morgan')
        features = encoder.encode_batch(processed['processed_smiles'])
        
        assert features.shape[0] == len(processed['processed_smiles'])
```

### Benchmark Test Example

```python
import pytest
import time

@pytest.mark.benchmark
class TestPerformance:
    def test_encoding_performance(self, benchmark):
        """Benchmark encoding performance."""
        smiles = ['CCO'] * 1000
        
        def encode_batch():
            encoder = get_encoder('morgan')
            return encoder.encode_batch(smiles)
        
        result = benchmark(encode_batch)
        assert len(result) == 1000
```

## Mocking Dependencies

Since molenc has optional dependencies (RDKit, PyTorch, etc.), tests use mocking extensively:

### RDKit Mocking Pattern

```python
from unittest.mock import patch, MagicMock

def test_with_rdkit_mock():
    with patch('molenc.encoders.fingerprints.rdkit') as mock_rdkit:
        # Setup mock
        mock_mol = MagicMock()
        mock_rdkit.Chem.MolFromSmiles.return_value = mock_mol
        mock_rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect.return_value = [1, 0] * 1024
        
        # Test code that uses RDKit
        encoder = MorganEncoder()
        result = encoder.encode('CCO')
        
        # Assertions
        assert result is not None
        mock_rdkit.Chem.MolFromSmiles.assert_called_with('CCO')
```

### PyTorch Mocking Pattern

```python
def test_with_torch_mock():
    with patch('molenc.encoders.representations.torch') as mock_torch:
        # Setup mock
        mock_tensor = MagicMock()
        mock_torch.tensor.return_value = mock_tensor
        
        # Test code
        encoder = TransformerEncoder()
        result = encoder.encode('CCO')
        
        assert result is not None
```

## Continuous Integration

### GitHub Actions Workflow

The `.github/workflows/ci.yml` file defines the CI pipeline:

- **Multi-Python Testing**: Tests on Python 3.8, 3.9, 3.10, 3.11
- **Dependency Testing**: Tests with different optional dependency groups
- **Coverage Reporting**: Uploads coverage to Codecov
- **Benchmark Tracking**: Tracks performance over time
- **Documentation Building**: Builds and deploys docs

### Running CI Locally

```bash
# Install act (GitHub Actions local runner)
# https://github.com/nektos/act

# Run CI workflow locally
act -j test
```

## Coverage Reports

### Generating Coverage

```bash
# Generate HTML coverage report
pytest tests/ --cov=molenc --cov-report=html

# Generate XML coverage report (for CI)
pytest tests/ --cov=molenc --cov-report=xml

# Generate terminal coverage report
pytest tests/ --cov=molenc --cov-report=term-missing
```

### Coverage Targets

- **Overall Coverage**: ≥80%
- **Core Modules**: ≥90%
- **Encoders**: ≥85%
- **Preprocessing**: ≥85%
- **Environment Management**: ≥80%

### Viewing Coverage

```bash
# Open HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Performance Testing

### Benchmark Tests

Benchmark tests measure performance characteristics:

```bash
# Run benchmark tests
pytest tests/benchmarks/ --benchmark-only

# Save benchmark results
pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark.json

# Compare benchmarks
pytest tests/benchmarks/ --benchmark-compare=benchmark.json
```

### Performance Metrics

- **Throughput**: Molecules processed per second
- **Memory Usage**: Peak memory consumption
- **Scalability**: Performance vs. dataset size
- **Latency**: Time per individual operation

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install in development mode
   pip install -e .
   ```

2. **Missing Dependencies**
   ```bash
   # Install test dependencies
   pip install -e .[dev]
   ```

3. **RDKit Import Errors**
   ```bash
   # Tests should work without RDKit (using mocks)
   # If you need RDKit for integration tests:
   conda install -c conda-forge rdkit
   ```

4. **Coverage Issues**
   ```bash
   # Install coverage
   pip install coverage pytest-cov
   ```

### Debug Mode

```bash
# Run tests with debug output
pytest tests/ -v -s --tb=long

# Run specific test with debugging
pytest tests/core/test_base.py::TestBaseEncoder::test_method -v -s
```

### Test Data

Test data is generated programmatically or uses small, well-known molecules:

```python
# Common test molecules
TEST_SMILES = [
    'CCO',          # Ethanol
    'CC(=O)O',      # Acetic acid
    'c1ccccc1',     # Benzene
    'CCN(CC)CC',    # Triethylamine
]
```

## Best Practices

### Test Organization

1. **One test class per module/class being tested**
2. **Descriptive test names**: `test_method_condition_expected_result`
3. **Use fixtures for common setup**
4. **Mock external dependencies**
5. **Test both success and failure cases**

### Test Quality

1. **Fast tests**: Unit tests should run in milliseconds
2. **Isolated tests**: No dependencies between tests
3. **Deterministic tests**: Same input → same output
4. **Clear assertions**: Test one thing at a time
5. **Good coverage**: Test edge cases and error conditions

### Documentation

1. **Docstrings for test classes and complex tests**
2. **Comments for non-obvious test logic**
3. **README for test-specific setup**

## Contributing

When contributing new features:

1. **Write tests first** (TDD approach)
2. **Maintain coverage** (≥80% overall)
3. **Add appropriate markers**
4. **Update this documentation** if needed
5. **Run full test suite** before submitting PR

```bash
# Pre-commit checklist
python run_tests.py --all --coverage
python run_tests.py --benchmark
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [pytest-benchmark Documentation](https://pytest-benchmark.readthedocs.io/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)