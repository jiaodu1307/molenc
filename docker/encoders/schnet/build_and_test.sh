#!/bin/bash

# SchNet Encoder Build and Test Script
# This script builds the SchNet Docker image and runs tests

set -e

echo "==================================="
echo "SchNet Encoder Build and Test"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_status "Docker is running"
}

# Function to build the SchNet image
build_schnet() {
    print_status "Building SchNet Docker image..."
    
    if docker build -f docker/encoders/schnet/Dockerfile -t molenc-schnet:latest .; then
        print_status "SchNet Docker image built successfully"
    else
        print_error "Failed to build SchNet Docker image"
        exit 1
    fi
}

# Function to start the SchNet service
start_schnet() {
    print_status "Starting SchNet service..."
    
    cd docker/compose
    if docker-compose up -d schnet; then
        print_status "SchNet service started"
    else
        print_error "Failed to start SchNet service"
        exit 1
    fi
    cd ../..
}

# Function to wait for service to be ready
wait_for_service() {
    print_status "Waiting for SchNet service to be ready..."
    
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8004/health >/dev/null 2>&1; then
            print_status "SchNet service is ready"
            return 0
        fi
        
        print_warning "Attempt $attempt/$max_attempts - Service not ready yet, waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    print_error "SchNet service did not become ready in time"
    return 1
}

# Function to run tests
run_tests() {
    print_status "Running SchNet tests..."
    
    cd docker/encoders/schnet
    if python test_schnet.py; then
        print_status "All tests passed"
        cd ../../..
        return 0
    else
        print_error "Some tests failed"
        cd ../../..
        return 1
    fi
}

# Function to test nginx proxy
test_nginx() {
    print_status "Testing nginx proxy..."
    
    if curl -f http://localhost/api/schnet/health >/dev/null 2>&1; then
        print_status "Nginx proxy is working correctly"
    else
        print_warning "Nginx proxy test failed - service might still be starting"
    fi
}

# Function to show service logs
show_logs() {
    print_status "Showing recent SchNet service logs..."
    docker-compose -f docker/compose/docker-compose.yml logs --tail=20 schnet
}

# Function to show service status
show_status() {
    print_status "SchNet service status:"
    docker-compose -f docker/compose/docker-compose.yml ps schnet
}

# Main execution
main() {
    print_status "Starting SchNet encoder build and test process..."
    
    # Check Docker
    check_docker
    
    # Build image
    build_schnet
    
    # Start service
    start_schnet
    
    # Wait for service to be ready
    if wait_for_service; then
        # Run tests
        if run_tests; then
            # Test nginx proxy
            test_nginx
            
            print_status "SchNet encoder setup completed successfully!"
            print_status "Service is available at:"
            print_status "  - Direct: http://localhost:8004"
            print_status "  - Proxy:  http://localhost/api/schnet/"
            
            show_status
        else
            print_error "Tests failed - check logs below:"
            show_logs
            exit 1
        fi
    else
        print_error "Service failed to start properly - check logs below:"
        show_logs
        exit 1
    fi
}

# Handle command line arguments
case "${1:-}" in
    "build")
        check_docker
        build_schnet
        ;;
    "start")
        start_schnet
        wait_for_service
        show_status
        ;;
    "test")
        run_tests
        ;;
    "stop")
        print_status "Stopping SchNet service..."
        cd docker/compose
        docker-compose stop schnet
        cd ../..
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "")
        main
        ;;
    *)
        echo "Usage: $0 [build|start|test|stop|logs|status]"
        echo ""
        echo "Commands:"
        echo "  build  - Build the Docker image"
        echo "  start  - Start the service"
        echo "  test   - Run tests"
        echo "  stop   - Stop the service"
        echo "  logs   - Show service logs"
        echo "  status - Show service status"
        echo "  (no argument) - Full build and test process"
        exit 1
        ;;
esac