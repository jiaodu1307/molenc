# MACCS Keys Encoder Docker Configuration

This directory contains the Docker configuration for the MACCS (Molecular ACCess System) keys fingerprint encoder.

## Overview

MACCS keys are a set of 167 structural features that capture important molecular characteristics for similarity searching and machine learning applications. This encoder provides a fast and reliable way to generate MACCS fingerprints for molecular datasets.

## Files

- `maccs_encoder.py` - Core MACCS encoder implementation
- `app.py` - FastAPI web service
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image configuration

## Features

- **167-bit binary fingerprint**: Standard MACCS keys implementation
- **Fast processing**: Optimized for batch processing of large datasets
- **Error handling**: Robust validation of SMILES strings
- **REST API**: Standardized endpoints following the MolEnc pattern
- **Batch support**: Efficient processing of large molecule sets

## API Endpoints

### Health Check
```bash
curl http://localhost:8003/health
# or through gateway
curl http://localhost/api/maccs/health
```

### Encoder Information
```bash
curl http://localhost:8003/info
```

### Single Molecule Encoding
```bash
curl -X POST http://localhost:8003/encode \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'
```

### Batch Encoding
```bash
curl -X POST http://localhost:8003/encode/batch \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["CCO", "CCCO", "CCCCO"]}'
```

## Building

To build the MACCS encoder Docker image:

```bash
cd /path/to/molenc
docker build -f docker/encoders/maccs/Dockerfile -t molenc-maccs:latest .
```

## Running

The MACCS encoder can be run as part of the full MolEnc Docker stack:

```bash
cd docker/compose
docker-compose up -d
```

Or run standalone:

```bash
docker run -p 8003:8000 molenc-maccs:latest
```

## Testing

Run the test script to verify the encoder is working correctly:

```bash
python docker/examples/test_maccs.py
```

## Usage Examples

See `docker/examples/maccs_example.py` for comprehensive usage examples including:
- Basic molecule encoding
- Drug molecule analysis
- Similarity calculations
- Performance testing

## Performance Characteristics

- **Processing speed**: ~5000+ molecules/second (depending on hardware)
- **Memory usage**: Low memory footprint
- **Batch size**: Optimized for 5000 molecules per batch
- **Fingerprint size**: Fixed 167-bit binary fingerprint

## Integration

The MACCS encoder integrates seamlessly with the MolEnc ecosystem:
- Uses the same API patterns as other encoders
- Routes through nginx gateway at `/api/maccs/`
- Supports the same request/response formats
- Participates in health checks and monitoring

## Dependencies

- RDKit (included in base image)
- FastAPI
- NumPy
- Standard MolEnc base dependencies

## Notes

- MACCS keys are predefined structural patterns
- The fingerprint includes 167 binary features
- Bit 0 is always 0 (reserved)
- Invalid SMILES are rejected with appropriate error messages
- Batch processing is recommended for large datasets