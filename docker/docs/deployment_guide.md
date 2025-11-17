# MolEnc Docker 部署指南

本指南详细介绍了如何部署和运行Docker化的分子编码器服务。

## 系统要求

### 硬件要求
- **CPU**: 多核处理器（推荐4核以上）
- **内存**: 最少8GB RAM（推荐16GB以上）
- **存储**: 最少20GB可用磁盘空间
- **GPU** (可选): NVIDIA GPU（用于加速ChemBERTa编码器）

### 软件要求
- **Docker**: 20.10.0 或更高版本
- **Docker Compose**: 1.29.0 或更高版本
- **操作系统**: Linux, macOS, 或 Windows (WSL2)

## 快速部署

### 1. 克隆项目
```bash
git clone <repository-url>
cd molenc
```

### 2. 快速启动
```bash
# 使用快速启动脚本
./docker/scripts/quickstart.sh
```

该脚本会自动：
- 检查Docker环境
- 创建必要的目录
- 构建Docker镜像
- 启动服务
- 进行健康检查

### 3. 验证部署
```bash
# 测试健康检查
curl http://localhost/health

# 测试Morgan编码器
curl -X POST http://localhost/api/morgan/encode \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["CCO", "c1ccccc1"]}'

# 测试ChemBERTa编码器
curl -X POST http://localhost/api/chemberta/encode \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["CCO", "c1ccccc1"]}'
```

## 手动部署

### 1. 构建Docker镜像

#### 构建基础镜像
```bash
docker build -t molenc-base:latest -f docker/base/Dockerfile.base .
```

#### 构建编码器镜像
```bash
# Morgan编码器
docker build -t molenc-morgan:latest -f docker/encoders/morgan/Dockerfile .

# ChemBERTa编码器
docker build -t molenc-chemberta:latest -f docker/encoders/chemberta/Dockerfile .
```

#### 构建网关镜像
```bash
docker build -t molenc-gateway:latest -f docker/gateway/Dockerfile .
```

### 2. 配置环境

创建必要的目录：
```bash
mkdir -p cache models logs
```

### 3. 启动服务

使用Docker Compose：
```bash
cd docker/compose
docker-compose up -d
```

或者手动启动：
```bash
# 启动Morgan编码器
docker run -d --name molenc-morgan \
  -p 8001:8000 \
  --network molenc-network \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  molenc-morgan:latest

# 启动ChemBERTa编码器
docker run -d --name molenc-chemberta \
  -p 8002:8000 \
  --network molenc-network \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  molenc-chemberta:latest

# 启动Nginx网关
docker run -d --name molenc-gateway \
  -p 80:80 -p 8080:8080 \
  --network molenc-network \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf \
  molenc-gateway:latest
```

## GPU支持

### NVIDIA GPU支持

确保已安装NVIDIA Docker运行时：
```bash
# 检查NVIDIA Docker运行时
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

在docker-compose.yml中启用GPU支持：
```yaml
services:
  chemberta:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 配置选项

### 环境变量

#### Morgan编码器
- `N_BITS`: 指纹位数 (默认: 1024)
- `RADIUS`: 指纹半径 (默认: 2)
- `USE_FEATURES`: 使用特征 (默认: false)

#### ChemBERTa编码器
- `MODEL_NAME`: 模型名称 (默认: seyonec/ChemBERTa-zinc-base-v1)
- `MAX_LENGTH`: 最大序列长度 (默认: 512)
- `POOLING_STRATEGY`: 池化策略 (默认: mean)
- `DEVICE`: 设备类型 (默认: auto)

#### Nginx网关
- `WORKER_PROCESSES`: 工作进程数 (默认: auto)
- `WORKER_CONNECTIONS`: 每个工作进程连接数 (默认: 1024)
- `CLIENT_MAX_BODY_SIZE`: 最大请求体大小 (默认: 10m)
- `PROXY_TIMEOUT`: 代理超时时间 (默认: 300s)

### 网络配置

默认网络配置：
```yaml
networks:
  molenc-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## 监控和日志

### 日志查看
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f morgan
docker-compose logs -f chemberta
docker-compose logs -f nginx
```

### 健康检查

服务提供健康检查端点：
```bash
# 网关健康检查
curl http://localhost/health

# Morgan编码器健康检查
curl http://localhost/api/morgan/health

# ChemBERTa编码器健康检查
curl http://localhost/api/chemberta/health
```

### 性能监控

使用内置的API测试工具：
```bash
# 运行性能测试
python docker/examples/api_test.py --concurrent 10 --load 30

# 生成详细报告
python docker/examples/api_test.py --concurrent 20 --load 60 --output test_report.json
```

## 扩展部署

### 水平扩展

#### 扩展Morgan编码器
```yaml
services:
  morgan:
    image: molenc-morgan:latest
    deploy:
      replicas: 3
```

#### 扩展ChemBERTa编码器
```yaml
services:
  chemberta:
    image: molenc-chemberta:latest
    deploy:
      replicas: 2
    environment:
      - DEVICE=auto
```

### 负载均衡

Nginx自动处理负载均衡：
```nginx
upstream morgan_backend {
    server morgan1:8000;
    server morgan2:8000;
    server morgan3:8000;
}
```

## 安全配置

### 防火墙规则
```bash
# 只允许必要端口
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8080/tcp
```

### SSL/TLS配置
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/ssl/certs/molenc.crt;
    ssl_certificate_key /etc/ssl/private/molenc.key;
    
    location / {
        proxy_pass http://backend;
        proxy_ssl_verify on;
    }
}
```

## 故障排除

### 常见问题

#### 1. Docker镜像构建失败
```bash
# 检查Dockerfile语法
docker build --no-cache -t test:latest -f docker/base/Dockerfile.base .

# 查看构建日志
docker build --progress=plain -t molenc-base:latest -f docker/base/Dockerfile.base .
```

#### 2. 服务无法启动
```bash
# 检查端口占用
netstat -tulpn | grep -E ':(80|8001|8002|8080)'

# 检查Docker容器状态
docker ps -a
docker logs <container_name>
```

#### 3. GPU不可用
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 检查容器日志
docker logs molenc-chemberta
```

#### 4. 内存不足
```bash
# 检查内存使用
free -h
docker stats

# 调整ChemBERTa批处理大小
docker run -e BATCH_SIZE=16 molenc-chemberta:latest
```

### 调试工具

#### 容器内调试
```bash
# 进入容器
docker exec -it molenc-morgan bash
docker exec -it molenc-chemberta bash

# 检查进程
ps aux

# 检查端口监听
netstat -tulpn
```

#### 网络调试
```bash
# 检查网络连接
docker network ls
docker network inspect molenc-network

# 测试容器间通信
docker exec molenc-gateway curl http://morgan:8000/health
docker exec molenc-gateway curl http://chemberta:8000/health
```

## 维护和更新

### 更新服务
```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose down
docker-compose up -d
```

### 备份和恢复

#### 备份配置
```bash
# 备份配置文件
tar -czf molenc-config-backup.tar.gz docker/ nginx.conf docker-compose.yml
```

#### 备份数据
```bash
# 备份缓存和模型
tar -czf molenc-data-backup.tar.gz cache models logs
```

### 清理资源
```bash
# 清理未使用的镜像
docker image prune -a

# 清理未使用的卷
docker volume prune

# 清理日志
docker-compose logs --tail=100 > logs_backup.txt
echo "" > $(docker inspect -f '{{.LogPath}}' molenc-morgan)
echo "" > $(docker inspect -f '{{.LogPath}}' molenc-chemberta)
```

## 性能优化

### Morgan编码器优化
- 调整指纹参数：`N_BITS=2048`, `RADIUS=3`
- 启用特征：`USE_FEATURES=true`
- 使用多进程：增加工作进程数

### ChemBERTa编码器优化
- 使用GPU：`DEVICE=cuda`
- 调整批处理大小：`BATCH_SIZE=64`
- 模型量化：使用INT8量化
- 模型缓存：启用模型缓存

### Nginx优化
- 调整工作进程：`worker_processes auto`
- 启用缓存：proxy_cache
- 调整超时：proxy_timeout
- 启用压缩：gzip

## 支持

### 获取帮助
- 查看日志：`docker-compose logs`
- 运行测试：`python docker/examples/api_test.py`
- 检查健康状态：`curl http://localhost/health`

### 报告问题
- 收集日志：`docker-compose logs > issue_logs.txt`
- 收集配置：`tar -czf config.tar.gz docker/`
- 创建问题报告：包含错误信息、配置、日志

## 版本历史

- v1.0.0: 初始版本，支持Morgan和ChemBERTa编码器
- v1.1.0: 添加GPU支持，优化性能
- v1.2.0: 添加负载均衡，支持水平扩展