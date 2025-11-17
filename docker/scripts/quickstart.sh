#!/bin/bash
set -e

# MolEnc Docker快速启动脚本

echo "🧬 MolEnc Docker化分子编码器快速启动"
echo "======================================"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建必要目录
echo "📁 创建必要目录..."
mkdir -p cache models logs

# 设置权限
chmod 755 cache models logs

# 检查端口是否被占用
check_port() {
    if netstat -tulpn 2>/dev/null | grep -q ":$1 "; then
        echo "❌ 端口 $1 已被占用"
        return 1
    fi
    return 0
}

echo "🔍 检查端口..."
check_port 80 || { echo "请修改docker-compose.yml中的端口配置"; exit 1; }
check_port 8001 || { echo "请修改docker-compose.yml中的端口配置"; exit 1; }
check_port 8002 || { echo "请修改docker-compose.yml中的端口配置"; exit 1; }

# 构建镜像
echo "🔨 构建Docker镜像..."
cd ../compose
docker-compose build --no-cache

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 健康检查
echo "🏥 进行健康检查..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost/health &>/dev/null; then
        echo "✅ 网关服务正常"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  重试 $RETRY_COUNT/$MAX_RETRIES..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "❌ 服务启动失败，请检查日志"
    echo "📋 查看日志命令:"
    echo "  docker-compose logs morgan"
    echo "  docker-compose logs chemberta"
    echo "  docker-compose logs nginx"
    exit 1
fi

# 测试各个服务
echo "🧪 测试各个服务..."

# 测试Morgan服务
if curl -f http://localhost:8001/health &>/dev/null; then
    echo "✅ Morgan服务正常"
else
    echo "❌ Morgan服务异常"
fi

# 测试ChemBERTa服务
if curl -f http://localhost:8002/health &>/dev/null; then
    echo "✅ ChemBERTa服务正常"
else
    echo "❌ ChemBERTa服务异常"
fi

# 显示状态信息
echo ""
echo "🎉 MolEnc Docker服务启动成功！"
echo "======================================"
echo "服务地址："
echo "  🔗 网关: http://localhost"
echo "  🔗 Morgan: http://localhost:8001"
echo "  🔗 ChemBERTa: http://localhost:8002"
echo "  🔗 管理界面: http://localhost:8080"
echo ""
echo "API示例："
echo "  # Morgan指纹编码"
echo "  curl -X POST http://localhost/api/morgan/encode \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"smiles\": \"CCO\", \"n_bits\": 2048}'"
echo ""
echo "  # ChemBERTa编码"
echo "  curl -X POST http://localhost/api/chemberta/encode \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"smiles\": \"CCO\", \"pooling_strategy\": \"mean\"}'"
echo ""
echo "管理命令："
echo "  # 查看状态"
echo "  docker-compose ps"
echo ""
echo "  # 查看日志"
echo "  docker-compose logs -f [service_name]"
echo ""
echo "  # 停止服务"
echo "  docker-compose down"
echo ""
echo "  # 重新启动"
echo "  docker-compose restart"
echo ""
echo "📁 数据目录："
echo "  cache/: 缓存文件"
echo "  models/: 模型文件"
echo "  logs/: 日志文件"
echo ""
echo "如需帮助，请查看文档或提交Issue"