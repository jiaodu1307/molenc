#!/bin/bash
# Build script for MACCS encoder Docker image

set -e

echo "=== Building MACCS Encoder Docker Image ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="molenc-maccs"
IMAGE_TAG="latest"
CONTEXT_DIR="../.."
DOCKERFILE="docker/encoders/maccs/Dockerfile"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if base image exists
if ! docker image inspect molenc-base:latest > /dev/null 2>&1; then
    print_warning "Base image 'molenc-base:latest' not found. Building it first..."
    cd "$(dirname "$0")"
    ./build_base.sh
fi

print_status "Building MACCS encoder image..."
print_status "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
print_status "Context: ${CONTEXT_DIR}"
print_status "Dockerfile: ${DOCKERFILE}"

# Build the image
if docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${DOCKERFILE}" \
    "${CONTEXT_DIR}"; then
    
    print_status "Successfully built ${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Display image info
    echo ""
    print_status "Image details:"
    docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    echo ""
    print_status "Build completed successfully!"
    print_status "You can now run the MACCS encoder with:"
    echo -e "  ${YELLOW}docker run -p 8003:8000 ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
    echo ""
    print_status "Or start the full stack with:"
    echo -e "  ${YELLOW}cd docker/compose && docker-compose up -d${NC}"
    
else
    print_error "Failed to build ${IMAGE_NAME}:${IMAGE_TAG}"
    exit 1
fi