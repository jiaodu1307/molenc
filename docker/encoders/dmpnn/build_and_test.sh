#!/bin/bash
# D-MPNN Encoder Build and Test Script
# Builds the D-MPNN Docker image and runs comprehensive tests

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_CONTEXT="$PROJECT_ROOT"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"
IMAGE_NAME="molenc-dmpnn:latest"
CONTAINER_NAME="molenc-dmpnn"
SERVICE_PORT=8005

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    log_info "Checking Docker status..."
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    log_success "Docker is running"
}

# Build Docker image
build_image() {
    log_info "Building D-MPNN Docker image..."
    
    cd "$DOCKER_CONTEXT"
    
    if docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" .; then
        log_success "Docker image built successfully: $IMAGE_NAME"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Run container
run_container() {
    log_info "Starting D-MPNN container..."
    
    # Stop existing container if running
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_info "Stopping existing container..."
        docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true
    fi
    
    # Run new container
    if docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${SERVICE_PORT}:8000" \
        -e "MOLENC_CACHE_DIR=/app/cache" \
        -e "PYTHONUNBUFFERED=1" \
        --restart unless-stopped \
        "$IMAGE_NAME"; then
        log_success "Container started successfully"
    else
        log_error "Failed to start container"
        exit 1
    fi
}

# Wait for service to be ready
wait_for_service() {
    log_info "Waiting for service to be ready..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "http://localhost:${SERVICE_PORT}/health" > /dev/null 2>&1; then
            log_success "Service is ready"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - Service not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Service failed to become ready within timeout"
    return 1
}

# Run health check
test_health_check() {
    log_info "Testing health check endpoint..."
    
    response=$(curl -s "http://localhost:${SERVICE_PORT}/health")
    if echo "$response" | grep -q '"status"'; then
        log_success "Health check passed"
        echo "Response: $response"
    else
        log_error "Health check failed"
        return 1
    fi
}

# Test info endpoint
test_info_endpoint() {
    log_info "Testing info endpoint..."
    
    response=$(curl -s "http://localhost:${SERVICE_PORT}/info")
    if echo "$response" | grep -q '"name"'; then
        log_success "Info endpoint passed"
        echo "Response: $response"
    else
        log_error "Info endpoint failed"
        return 1
    fi
}

# Test single encoding
test_single_encoding() {
    log_info "Testing single molecule encoding..."
    
    response=$(curl -s -X POST "http://localhost:${SERVICE_PORT}/encode" \
        -H "Content-Type: application/json" \
        -d '{"smiles": "CCO", "depth": 3}')
    
    if echo "$response" | grep -q '"embeddings"'; then
        log_success "Single encoding passed"
        echo "Processing time: $(echo "$response" | grep -o '"processing_time":[0-9.]*' | cut -d: -f2) seconds"
    else
        log_error "Single encoding failed"
        echo "Response: $response"
        return 1
    fi
}

# Test batch encoding
test_batch_encoding() {
    log_info "Testing batch encoding..."
    
    response=$(curl -s -X POST "http://localhost:${SERVICE_PORT}/encode_batch" \
        -H "Content-Type: application/json" \
        -d '{"molecules": ["CCO", "CC(=O)O", "c1ccccc1"], "depth": 3}')
    
    if echo "$response" | grep -q '"embeddings"'; then
        log_success "Batch encoding passed"
        echo "Processing time: $(echo "$response" | grep -o '"processing_time":[0-9.]*' | cut -d: -f2) seconds"
    else
        log_error "Batch encoding failed"
        echo "Response: $response"
        return 1
    fi
}

# Test stream encoding
test_stream_encoding() {
    log_info "Testing stream encoding..."
    
    response=$(curl -s -X POST "http://localhost:${SERVICE_PORT}/encode_stream" \
        -H "Content-Type: application/json" \
        -d '{"molecules": ["CCO", "CC(=O)O", "c1ccccc1"], "depth": 3}')
    
    if echo "$response" | grep -q '"results"'; then
        log_success "Stream encoding passed"
        successful=$(echo "$response" | grep -o '"successful":[0-9]*' | cut -d: -f2)
        failed=$(echo "$response" | grep -o '"failed":[0-9]*' | cut -d: -f2)
        echo "Successful: $successful, Failed: $failed"
    else
        log_error "Stream encoding failed"
        echo "Response: $response"
        return 1
    fi
}

# Test error handling
test_error_handling() {
    log_info "Testing error handling..."
    
    # Test invalid SMILES
    response=$(curl -s -X POST "http://localhost:${SERVICE_PORT}/encode" \
        -H "Content-Type: application/json" \
        -d '{"smiles": "invalid_smiles", "depth": 3}')
    
    if [ "$(echo "$response" | grep -c '"detail"')" -gt 0 ]; then
        log_success "Error handling for invalid SMILES passed"
    else
        log_error "Error handling for invalid SMILES failed"
        echo "Response: $response"
        return 1
    fi
}

# Run local tests
run_local_tests() {
    log_info "Running local tests..."
    
    cd "$SCRIPT_DIR"
    
    if python test_local.py; then
        log_success "Local tests passed"
    else
        log_error "Local tests failed"
        return 1
    fi
}

# Run API tests
run_api_tests() {
    log_info "Running API tests..."
    
    cd "$SCRIPT_DIR"
    
    if python test_dmpnn.py; then
        log_success "API tests passed"
    else
        log_error "API tests failed"
        return 1
    fi
}

# Performance test
performance_test() {
    log_info "Running performance test..."
    
    # Test with different batch sizes
    for batch_size in 1 10 50 100; do
        log_info "Testing batch size: $batch_size"
        
        # Generate test molecules
        molecules=$(python -c "
import json
molecules = ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN(CC)CC', 'CC(C)CCO'] * $((batch_size // 5 + 1))
print(json.dumps(molecules[:$batch_size]))
        ")
        
        start_time=$(date +%s.%N)
        response=$(curl -s -X POST "http://localhost:${SERVICE_PORT}/encode_batch" \
            -H "Content-Type: application/json" \
            -d "{\"molecules\": $molecules, \"depth\": 3}")
        end_time=$(date +%s.%N)
        
        if echo "$response" | grep -q '"embeddings"'; then
            processing_time=$(echo "$response" | grep -o '"processing_time":[0-9.]*' | cut -d: -f2)
            throughput=$(echo "scale=1; $batch_size / $processing_time" | bc -l 2>/dev/null || echo "N/A")
            log_success "Batch size $batch_size: ${processing_time}s, ${throughput} mol/s"
        else
            log_error "Performance test failed for batch size $batch_size"
        fi
    done
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Stop container
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || true
        log_success "Container stopped and removed"
    fi
}

# Main function
main() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "D-MPNN Encoder Build and Test Script"
    echo "=========================================="
    echo -e "${NC}"
    
    # Parse command line arguments
    case "${1:-all}" in
        build)
            check_docker
            build_image
            ;;
        run)
            check_docker
            run_container
            wait_for_service
            ;;
        test)
            check_docker
            run_container
            wait_for_service
            test_health_check
            test_info_endpoint
            test_single_encoding
            test_batch_encoding
            test_stream_encoding
            test_error_handling
            run_local_tests
            run_api_tests
            ;;
        performance)
            check_docker
            run_container
            wait_for_service
            performance_test
            ;;
        all)
            check_docker
            build_image
            run_container
            wait_for_service
            test_health_check
            test_info_endpoint
            test_single_encoding
            test_batch_encoding
            test_stream_encoding
            test_error_handling
            run_local_tests
            run_api_tests
            performance_test
            cleanup
            ;;
        cleanup)
            cleanup
            ;;
        *)
            echo "Usage: $0 {build|run|test|performance|all|cleanup}"
            echo ""
            echo "Commands:"
            echo "  build      - Build Docker image"
            echo "  run        - Run container"
            echo "  test       - Run tests"
            echo "  performance- Run performance tests"
            echo "  all        - Run all steps (default)"
            echo "  cleanup    - Clean up containers"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}"
    echo "=========================================="
    echo "Script completed successfully!"
    echo "=========================================="
    echo -e "${NC}"
}

# Set trap for cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"