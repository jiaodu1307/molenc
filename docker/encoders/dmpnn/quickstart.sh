#!/bin/bash
# Quick start script for D-MPNN encoder
# This script provides a quick way to start and test the D-MPNN service

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "======================================"
echo "D-MPNN Encoder Quick Start"
echo "======================================"
echo -e "${NC}"

# Change to the D-MPNN directory
cd "$(dirname "$0")"

echo "Building D-MPNN Docker image..."
docker build -t molenc-dmpnn:latest .

echo -e "${YELLOW}"
echo "Starting D-MPNN service..."
echo -e "${NC}"

# Run the container
docker run -d \
    --name molenc-dmpnn \
    -p 8005:8000 \
    -e "MOLENC_CACHE_DIR=/app/cache" \
    -e "PYTHONUNBUFFERED=1" \
    --restart unless-stopped \
    molenc-dmpnn:latest

echo "Waiting for service to start..."
sleep 10

echo -e "${GREEN}"
echo "Testing D-MPNN service..."
echo -e "${NC}"

# Test health endpoint
echo "Health check:"
curl -s http://localhost:8005/health | python -m json.tool

echo ""
echo "Info endpoint:"
curl -s http://localhost:8005/info | python -m json.tool

echo ""
echo "Testing single molecule encoding:"
curl -s -X POST http://localhost:8005/encode \
    -H "Content-Type: application/json" \
    -d '{"smiles": "CCO", "depth": 3}' | python -m json.tool

echo ""
echo -e "${GREEN}"
echo "D-MPNN service is running!"
echo "API endpoints:"
echo "  Health:    http://localhost:8005/health"
echo "  Info:      http://localhost:8005/info"
echo "  Encode:    http://localhost:8005/encode"
echo "  Batch:     http://localhost:8005/encode_batch"
echo "  Stream:    http://localhost:8005/encode_stream"
echo ""
echo "To stop the service: docker stop molenc-dmpnn"
echo "To remove the container: docker rm molenc-dmpnn"
echo -e "${NC}"