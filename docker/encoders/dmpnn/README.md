# D-MPNN Molecular Encoder

A Dockerized implementation of the Directed Message Passing Neural Network (D-MPNN) for molecular encoding, providing REST API endpoints for encoding SMILES strings into vector representations.

## Overview

D-MPNN is a graph neural network architecture specifically designed for molecular representation learning. It uses directed message passing on molecular graphs to learn meaningful representations that can be used for various downstream tasks such as property prediction, similarity search, and molecular generation.

## Features

- **Directed Message Passing**: Uses directed edges in molecular graphs for more informative message passing
- **REST API**: FastAPI-based service with multiple endpoints
- **Batch Processing**: Efficient batch encoding for multiple molecules
- **Stream Processing**: Real-time streaming for large datasets
- **Docker Support**: Fully containerized with health checks and monitoring
- **Nginx Integration**: Load balancing and reverse proxy support
- **Comprehensive Testing**: Unit tests and integration tests included

## Quick Start

### Using Docker Compose

1. Start the service:
```bash
docker-compose up -d dmpnn
```

2. Test the service:
```bash
# Health check
curl http://localhost:8005/health

# Encode a single molecule
curl -X POST http://localhost:8005/encode \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "depth": 3}'

# Encode multiple molecules
curl -X POST http://localhost:8005/encode_batch \
  -H "Content-Type: application/json" \
  -d '{"molecules": ["CCO", "CC(=O)O", "c1ccccc1"], "depth": 3}'
```

### Using Nginx Proxy

If using the nginx proxy (recommended for production):

```bash
# Health check through nginx
curl http://localhost/api/dmpnn/health

# Encode through nginx
curl -X POST http://localhost/api/dmpnn/encode \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "depth": 3}'
```

## API Endpoints

### Health Check
```http
GET /health
```

Returns service health status and encoder information.

### Encoder Information
```http
GET /info
```

Returns detailed information about the encoder configuration.

### Single Molecule Encoding
```http
POST /encode
Content-Type: application/json

{
  "smiles": "CCO",
  "depth": 3
}
```

Encodes a single SMILES string into a vector representation.

### Batch Encoding
```http
POST /encode_batch
Content-Type: application/json

{
  "molecules": ["CCO", "CC(=O)O", "c1ccccc1"],
  "depth": 3
}
```

Encodes multiple molecules in a single request.

### Stream Encoding
```http
POST /encode_stream
Content-Type: application/json

{
  "molecules": ["CCO", "CC(=O)O", "c1ccccc1"],
  "depth": 3
}
```

Encodes molecules with individual result streaming.

## Parameters

### Encoder Parameters

- `node_dim` (int): Dimension of node features (default: 64)
- `edge_dim` (int): Dimension of edge features (default: 64)
- `depth` (int): Number of message passing steps (default: 3)
- `dropout` (float): Dropout rate (default: 0.0)
- `aggregation` (str): Aggregation method ('mean', 'sum', 'max') (default: 'mean')

### API Parameters

- `smiles` (str): SMILES string to encode
- `molecules` (list): List of SMILES strings
- `depth` (int): Message passing depth (optional, default: 3)

## Installation

### Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t dmpnn-encoder .

# Run the container
docker run -p 8005:8005 dmpnn-encoder
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python app.py
```

## Testing

### Local Tests
```bash
python test_local.py
```

### API Tests
```bash
python test_dmpnn.py
```

### All Tests
```bash
# Run all tests with pytest
pytest test_*.py -v

# Or run the comprehensive test suite
python test_dmpnn.py
```

## Usage Examples

### Basic Usage
```python
from dmpnn_encoder import DMPNNEncoder

# Initialize encoder
encoder = DMPNNEncoder(
    node_dim=64,
    edge_dim=64,
    depth=3,
    dropout=0.0,
    aggregation='mean'
)

# Encode a single molecule
smiles = "CCO"  # Ethanol
embedding = encoder.encode_smiles(smiles)
print(f"Embedding shape: {embedding.shape}")
```

### Batch Encoding
```python
# Encode multiple molecules
molecules = ["CCO", "CC(=O)O", "c1ccccc1"]
embeddings = encoder.encode_batch(molecules)
print(f"Batch embeddings shape: {embeddings.shape}")
```

### Similarity Search
```python
import numpy as np

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Reference molecule
ref_smiles = "CCO"
ref_embedding = encoder.encode_smiles(ref_smiles)

# Candidate molecules
candidates = ["CCCO", "CC(C)O", "CC(=O)O"]
for smiles in candidates:
    embedding = encoder.encode_smiles(smiles)
    similarity = cosine_similarity(ref_embedding, embedding)
    print(f"{smiles}: {similarity:.3f}")
```

## Architecture

### D-MPNN Model
The D-MPNN encoder consists of:

1. **Atom Feature Extraction**: Converts RDKit atoms to feature vectors
2. **Bond Feature Extraction**: Converts RDKit bonds to feature vectors
3. **Directed Message Passing**: Propagates information through directed edges
4. **Readout Function**: Aggregates node features to molecule-level representation

### API Service
The FastAPI service provides:

- **Health Monitoring**: Service health and encoder status
- **Single Encoding**: Individual molecule encoding
- **Batch Processing**: Multiple molecule encoding
- **Stream Processing**: Real-time encoding with individual results
- **Error Handling**: Comprehensive error responses

## Performance

Typical performance metrics on modern hardware:

- **Single Molecule**: ~10-50ms
- **Batch Processing**: ~100-500 molecules/second
- **Memory Usage**: ~500MB-2GB depending on batch size
- **Vector Dimension**: Configurable (default: 64)

## Configuration

### Docker Configuration
The service runs on port 8005 by default and can be configured through environment variables:

- `PORT`: Service port (default: 8005)
- `WORKERS`: Number of worker processes (default: 1)
- `MAX_BATCH_SIZE`: Maximum batch size (default: 1000)

### Nginx Configuration
The nginx proxy provides:

- **Load Balancing**: Multiple encoder instances
- **Rate Limiting**: Request throttling
- **Health Checks**: Service monitoring
- **SSL Termination**: HTTPS support

## Troubleshooting

### Common Issues

1. **Service Not Starting**
   - Check Docker logs: `docker logs <container_id>`
   - Verify port availability: `netstat -an | grep 8005`

2. **Encoding Failures**
   - Validate SMILES strings
   - Check molecule size limits
   - Verify RDKit installation

3. **Performance Issues**
   - Adjust batch sizes
   - Monitor memory usage
   - Check CPU utilization

### Debug Mode
Enable debug logging:
```bash
export DEBUG=1
python app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this D-MPNN implementation in your research, please cite:

```bibtex
@article{yang2019analyzing,
  title={Analyzing learned molecular representations for property prediction},
  author={Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and Eiden, Philipp and Gao, Hua and Guzman-Perez, Angel and Hopper, Timothy and Kelley, Brian and Mathea, Miriam and others},
  journal={Journal of chemical information and modeling},
  volume={59},
  number={8},
  pages={3370--3388},
  year={2019},
  publisher={ACS Publications}
}
```

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the test files for usage examples