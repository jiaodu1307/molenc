#!/bin/bash
set -e

# MolEnc Docker构建脚本

echo "🧬 MolEnc Docker镜像构建脚本"
echo "======================================"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 默认参数
BUILD_BASE=${BUILD_BASE:-true}
BUILD_MORGAN=${BUILD_MORGAN:-true}
BUILD_CHEMBERTA=${BUILD_CHEMBERTA:-true}
PUSH_IMAGES=${PUSH_IMAGES:-false}
REGISTRY=${REGISTRY:-""}
TAG=${TAG:-"latest"}

echo "📋 构建参数:"
echo "  构建基础镜像: $BUILD_BASE"
echo "  构建Morgan: $BUILD_MORGAN"
echo "  构建ChemBERTa: $BUILD_CHEMBERTA"
echo "  推送镜像: $PUSH_IMAGES"
echo "  镜像标签: $TAG"
if [ -n "$REGISTRY" ]; then
    echo "  镜像仓库: $REGISTRY"
fi

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p "${PROJECT_ROOT}/docker/cache"
mkdir -p "${PROJECT_ROOT}/docker/models"
mkdir -p "${PROJECT_ROOT}/docker/logs"

# 构建基础镜像
if [ "$BUILD_BASE" = true ]; then
    echo "🔨 构建基础镜像..."
    cd "$PROJECT_ROOT"
    docker build -t "molenc-base:${TAG}" -f docker/base/Dockerfile.base .
    
    if [ "$PUSH_IMAGES" = true ] && [ -n "$REGISTRY" ]; then
        echo "📤 推送基础镜像..."
        docker tag "molenc-base:${TAG}" "${REGISTRY}/molenc-base:${TAG}"
        docker push "${REGISTRY}/molenc-base:${TAG}"
    fi
fi

# 构建Morgan编码器镜像
if [ "$BUILD_MORGAN" = true ]; then
    echo "🔨 构建Morgan编码器镜像..."
    cd "$PROJECT_ROOT"
    docker build -t "molenc-morgan:${TAG}" -f docker/encoders/morgan/Dockerfile .
    
    if [ "$PUSH_IMAGES" = true ] && [ -n "$REGISTRY" ]; then
        echo "📤 推送Morgan镜像..."
        docker tag "molenc-morgan:${TAG}" "${REGISTRY}/molenc-morgan:${TAG}"
        docker push "${REGISTRY}/molenc-morgan:${TAG}"
    fi
fi

# 构建ChemBERTa编码器镜像
if [ "$BUILD_CHEMBERTA" = true ]; then
    echo "🔨 构建ChemBERTa编码器镜像..."
    cd "$PROJECT_ROOT"
    docker build -t "molenc-chemberta:${TAG}" -f docker/encoders/chemberta/Dockerfile .
    
    if [ "$PUSH_IMAGES" = true ] && [ -n "$REGISTRY" ]; then
        echo "📤 推送ChemBERTa镜像..."
        docker tag "molenc-chemberta:${TAG}" "${REGISTRY}/molenc-chemberta:${TAG}"
        docker push "${REGISTRY}/molenc-chemberta:${TAG}"
    fi
fi

# 构建Nginx网关镜像（如果需要自定义）
echo "🔨 构建Nginx网关镜像..."
docker build -t "molenc-gateway:${TAG}" -f - . << 'EOF'
FROM nginx:alpine
COPY docker/nginx.conf /etc/nginx/nginx.conf
EXPOSE 80 8080
CMD ["nginx", "-g", "daemon off;"]
EOF

if [ "$PUSH_IMAGES" = true ] && [ -n "$REGISTRY" ]; then
    echo "📤 推送Nginx网关镜像..."
    docker tag "molenc-gateway:${TAG}" "${REGISTRY}/molenc-gateway:${TAG}"
    docker push "${REGISTRY}/molenc-gateway:${TAG}"
fi

echo ""
echo "✅ 构建完成！"
echo "======================================"
echo "镜像列表:"
docker images | grep -E "molenc|REPOSITORY"

echo ""
echo "🚀 下一步:"
echo "  1. 运行快速启动脚本: ./quickstart.sh"
echo "  2. 或手动启动: cd compose && docker-compose up -d"
echo ""
echo "📚 使用说明:"
echo "  API端点: http://localhost/api/{encoder}/"
echo "  管理界面: http://localhost:8080"
echo "  文档: docker/docs/"