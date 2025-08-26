# MolEnc Advanced Features Guide

This guide covers the advanced features implemented in MolEnc to provide intelligent encoder selection, dependency management, and enhanced usability.

## Overview

The advanced features include:

1. **Unified Encoder Factory** - Single interface for creating any encoder
2. **Intelligent Encoder Selection** - Automatic selection of optimal encoders
3. **Multi-Version Support** - Support for multiple versions of the same encoder
4. **Cloud API Integration** - Fallback to cloud-based encoding services
5. **Advanced Dependency Management** - Smart handling of optional dependencies
6. **Environment Isolation** - Run encoders in isolated environments
7. **Performance Comparison** - Benchmark and compare different encoders
8. **Precompiled Packages** - Pre-built encoder packages for easy installation

## Quick Start

### Basic Usage with Factory

```python
from molenc.core.encoder_factory import get_encoder_factory, EncoderMode

# Get the global factory instance
factory = get_encoder_factory()

# Create an encoder with smart selection
encoder = factory.create_encoder(
    'unimol',
    mode=EncoderMode.SMART,
    enable_cloud_fallback=True
)

# Encode molecules
embedding = encoder.encode_single("CCO")
batch_embeddings = encoder.encode_batch(["CCO", "CCN", "CCC"])
```

### Advanced UniMol Usage

```python
from molenc.encoders.representations.multimodal.unimol import UniMolEncoder

# Create UniMol encoder with advanced features
encoder = UniMolEncoder(
    model_name="unimol_v1",
    output_dim=512,
    enable_smart_selection=True,
    enable_cloud_fallback=True
)

# Check encoder status
status = encoder.get_status()
print(f"Model type: {status['model_type']}")
print(f"Advanced features: {status['advanced_features']}")

# Switch versions if available
if encoder.switch_version("HUGGINGFACE"):
    print("Switched to HuggingFace version")
```

## Feature Details

### 1. Unified Encoder Factory

The encoder factory provides a single interface for creating any type of molecular encoder:

```python
from molenc.core.encoder_factory import (
    get_encoder_factory, 
    EncoderMode, 
    EncoderConfig
)

factory = get_encoder_factory()

# List available encoders
available = factory.get_available_encoders()
print(available)

# Create encoder with different modes
encoder_basic = factory.create_encoder('unimol', mode=EncoderMode.BASIC)
encoder_smart = factory.create_encoder('unimol', mode=EncoderMode.SMART)
encoder_cloud = factory.create_encoder('unimol', mode=EncoderMode.CLOUD_ONLY)

# Create with custom configuration
config = EncoderConfig(
    output_dim=1024,
    use_3d=True,
    device='cuda'
)
encoder_custom = factory.create_encoder('unimol', config=config)
```

### 2. Intelligent Encoder Selection

The smart selector automatically chooses the best encoder configuration:

```python
from molenc.core.smart_encoder_selector import get_smart_selector

selector = get_smart_selector()

# Get optimal configuration for a molecule
config = selector.select_optimal_config(
    molecule_smiles="CC(C)(C)c1ccc(O)cc1",
    encoder_type='unimol',
    requirements={
        'speed': 'high',
        'accuracy': 'medium',
        'memory': 'low'
    }
)

# Register custom encoder variants
selector.register_encoder_variant(
    encoder_type='unimol',
    variant_name='fast_unimol',
    config={'output_dim': 256, 'use_3d': False},
    dependencies=['unimol_tools'],
    priority=0.8,
    performance_score=0.7,
    compatibility_score=0.9
)
```

### 3. Multi-Version Support

Support for multiple versions of the same encoder:

```python
from molenc.encoders.representations.multimodal.unimol_versions import (
    UniMolVersionManager,
    UniMolVersion,
    UniMolVersionInfo,
    get_version_manager
)

# Create multi-version encoder
encoder = MultiVersionUniMolEncoder(
    preferred_version=UniMolVersion.OFFICIAL_V1,
    fallback_versions=[UniMolVersion.HUGGINGFACE, UniMolVersion.PYTORCH]
)

# Check available versions
print(f"Active: {encoder.get_active_version()}")
print(f"Available: {encoder.get_available_versions()}")

# Switch versions
encoder.switch_version(UniMolVersion.HUGGINGFACE)
```

### 4. Cloud API Integration

Fallback to cloud-based encoding services:

```python
from molenc.cloud.api_client import (
    get_cloud_client,
    CloudProvider,
    CloudConfig
)

# Configure cloud client
config = CloudConfig(
    provider=CloudProvider.HUGGINGFACE,
    api_key="your-api-key",
    base_url="https://api.huggingface.co"
)

client = get_cloud_client(config)

# Use cloud encoding
result = client.encode_molecule(
    smiles="CCO",
    encoder_type="unimol",
    config={'output_dim': 512}
)

# Batch encoding
batch_result = client.encode_batch(
    smiles_list=["CCO", "CCN", "CCC"],
    encoder_type="unimol"
)
```

### 5. Advanced Dependency Management

Smart handling of optional dependencies:

```python
from molenc.core.advanced_dependency_manager import get_dependency_manager

manager = get_dependency_manager()

# Check encoder dependencies
status = manager.check_encoder_dependencies('unimol')
print(f"UniMol dependencies: {status}")

# Get installation suggestions
suggestions = manager.get_installation_suggestions('unimol')
for suggestion in suggestions:
    print(f"Install: {suggestion}")

# Install dependencies
manager.install_encoder_dependencies('unimol', strategy='progressive')
```

### 6. Environment Isolation

Run encoders in isolated environments:

```python
from molenc.isolation import (
    IsolationEnvironmentManager,
    ProcessWrapper,
    IsolationMethod
)
from molenc.isolation.environment_manager import (
    create_isolated_encoder,
    run_in_isolated_environment
)

manager = get_environment_manager()

# Create isolated environment
env_spec = manager.create_environment(
    name="unimol_env",
    method=IsolationMethod.VIRTUAL_ENV,
    encoder_type="unimol"
)

# Use environment
with manager.use_environment("unimol_env") as env:
    # Code runs in isolated environment
    encoder = env.create_encoder('unimol')
    result = encoder.encode_single("CCO")
```

### 7. Performance Comparison

Benchmark and compare different encoders:

```python
from molenc.benchmarks.performance_comparison import (
    get_benchmark,
    quick_compare_encoders,
    BenchmarkConfig
)

# Quick comparison
result = quick_compare_encoders(
    encoder_types=['unimol', 'morgan', 'rdkit'],
    test_molecules=["CCO", "CCN", "CCC"],
    metrics=['speed', 'memory', 'accuracy']
)

print(result.summary)
print(result.recommendations)

# Detailed benchmarking
benchmark = get_benchmark()
config = BenchmarkConfig(
    test_molecules=["CCO"] * 100,
    batch_sizes=[1, 10, 50],
    metrics=['encoding_speed', 'memory_usage', 'initialization_time']
)

results = benchmark.run_comparison(['unimol', 'morgan'], config)
```

### 8. Precompiled Packages

Use pre-built encoder packages:

```python
from molenc.precompiled.package_manager import (
    PrecompiledPackageManager,
    PackageSpec,
    Platform,
    PackageType,
    install_encoder_package,
    is_encoder_available
)

# Check availability
if not is_encoder_available('unimol'):
    # Install precompiled package
    install_encoder_package('unimol', version='latest')

# List installed packages
manager = get_package_manager()
packages = manager.list_installed_packages()
for package in packages:
    print(f"{package.name} v{package.version}")
```

## Configuration

### Global Configuration

Configure global settings for advanced features:

```python
from molenc.core.config import set_global_config

set_global_config({
    'enable_smart_selection': True,
    'enable_cloud_fallback': True,
    'cloud_provider': 'huggingface',
    'cache_encoders': True,
    'isolation_method': 'virtual_env',
    'benchmark_cache': True
})
```

### Environment Variables

Set environment variables for configuration:

```bash
export MOLENC_ENABLE_SMART_SELECTION=true
export MOLENC_CLOUD_API_KEY=your-api-key
export MOLENC_CACHE_DIR=/path/to/cache
export MOLENC_ISOLATION_METHOD=docker
```

## Error Handling

The advanced features include comprehensive error handling:

```python
from molenc.core.exceptions import (
    EncoderNotFoundError,
    DependencyError,
    CloudAPIError,
    IsolationError
)

try:
    encoder = factory.create_encoder('nonexistent_encoder')
except EncoderNotFoundError as e:
    print(f"Encoder not found: {e}")

try:
    encoder = factory.create_encoder('unimol', mode=EncoderMode.CLOUD_ONLY)
except CloudAPIError as e:
    print(f"Cloud API error: {e}")
```

## Best Practices

1. **Use the Factory**: Always use the encoder factory for creating encoders
2. **Enable Smart Selection**: Let the system choose optimal configurations
3. **Configure Cloud Fallback**: Ensure reliability with cloud backup
4. **Monitor Performance**: Use benchmarking to optimize your workflow
5. **Isolate Environments**: Use isolation for production deployments
6. **Cache Results**: Enable caching for repeated operations
7. **Handle Errors**: Implement proper error handling for robustness

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Cloud API Failures**: Check API keys and network connectivity
3. **Version Conflicts**: Use environment isolation
4. **Performance Issues**: Use benchmarking to identify bottlenecks
5. **Memory Problems**: Adjust batch sizes and enable memory monitoring

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
# export MOLENC_LOG_LEVEL=DEBUG
```

## Examples

See the `examples/` directory for complete usage examples:

- `advanced_features_demo.py` - Comprehensive demo of all features
- `smart_selection_example.py` - Intelligent encoder selection
- `cloud_integration_example.py` - Cloud API usage
- `performance_comparison_example.py` - Benchmarking examples
- `isolation_example.py` - Environment isolation usage

## API Reference

For detailed API documentation, see:

- `molenc.core.encoder_factory` - Unified encoder factory
- `molenc.core.smart_encoder_selector` - Intelligent selection
- `molenc.core.advanced_dependency_manager` - Dependency management
- `molenc.cloud.api_client` - Cloud API integration
- `molenc.isolation.environment_manager` - Environment isolation
- `molenc.benchmarks.performance_comparison` - Performance tools
- `molenc.precompiled.package_manager` - Package management