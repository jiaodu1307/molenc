# SchNet Molecular Encoder

SchNet分子编码器的Docker化实现，基于连续滤波卷积神经网络处理3D分子结构。

## 概述

SchNet是一种基于3D结构的分子编码器，使用连续滤波卷积神经网络将分子几何结构转换为固定大小的向量表示。该实现支持：

- 3D分子结构编码
- 批量分子处理
- GPU加速（如果可用）
- RESTful API接口
- Docker容器化部署

## 架构

### 核心组件

1. **SchNetEncoder**: 主要的编码器类，实现SchNet架构
2. **InteractionBlock**: SchNet交互块，实现连续滤波卷积
3. **GaussianBasis**: 高斯基函数用于距离编码
4. **FastAPI Service**: RESTful API服务接口

### 技术栈

- **PyTorch**: 深度学习框架
- **RDKit**: 化学信息学库
- **FastAPI**: Web服务框架
- **Docker**: 容器化部署
- **Nginx**: 负载均衡和路由

## 快速开始

### Docker部署

1. 构建和启动服务：
```bash
cd /home/jiaodu/projects/molenc
docker compose up -d schnet
```

2. 测试服务：
```bash
# 健康检查
curl http://localhost:8004/health

# 编码单个分子
curl -X POST http://localhost:8004/encode \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "max_conformers": 1}'

# 批量编码
curl -X POST http://localhost:8004/encode_batch \
  -H "Content-Type: application/json" \
  -d '{"smiles_list": ["CCO", "c1ccccc1"], "max_conformers": 1}'
```

### 本地测试

如果Docker环境不可用，可以使用本地测试脚本：

```bash
cd /home/jiaodu/projects/molenc/docker/encoders/schnet
python test_local.py
```

## API接口

### 端点

#### 1. 健康检查
```
GET /health
```
返回服务状态信息。

#### 2. 编码器信息
```
GET /info
```
返回编码器的配置信息。

#### 3. 单分子编码
```
POST /encode
```

请求体：
```json
{
  "smiles": "CCO",
  "max_conformers": 1
}
```

响应：
```json
{
  "embedding": [0.1, 0.2, 0.3, ...],
  "smiles": "CCO",
  "embedding_dim": 128
}
```

#### 4. 批量编码
```
POST /encode_batch
```

请求体：
```json
{
  "smiles_list": ["CCO", "c1ccccc1"],
  "max_conformers": 1
}
```

响应：
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "smiles_list": ["CCO", "c1ccccc1"],
  "embedding_dim": 128
}
```

#### 5. 流式编码
```
POST /encode_stream
```

支持大分子的流式处理。

## 配置参数

### 模型参数

- `hidden_channels`: 隐藏层通道数 (默认: 128)
- `num_filters`: 滤波器数量 (默认: 128)
- `num_interactions`: 交互块数量 (默认: 6)
- `num_gaussians`: 高斯基函数数量 (默认: 50)
- `cutoff`: 截止距离 (默认: 10.0 Å)

### 服务参数

- `host`: 服务主机 (默认: 0.0.0.0)
- `port`: 服务端口 (默认: 8000)
- `workers`: 工作进程数 (默认: 1)
- `timeout`: 请求超时时间 (默认: 120s)

## 使用示例

### Python客户端

```python
import requests
import json

# API端点
base_url = "http://localhost:8004"

# 编码单个分子
response = requests.post(
    f"{base_url}/encode",
    json={"smiles": "CCO", "max_conformers": 1}
)
result = response.json()
print(f"Embedding shape: {len(result['embedding'])}")

# 批量编码
response = requests.post(
    f"{base_url}/encode_batch",
    json={"smiles_list": ["CCO", "c1ccccc1", "CC(=O)O"]}
)
result = response.json()
print(f"Batch embeddings shape: {len(result['embeddings'])}")
```

### 直接调用编码器

```python
from schnet_encoder import SchNetEncoder

# 初始化编码器
encoder = SchNetEncoder()

# 编码单个分子
embedding = encoder.encode_smiles("CCO")
print(f"Embedding shape: {embedding.shape}")

# 批量编码
embeddings = encoder.encode_batch(["CCO", "c1ccccc1"])
print(f"Batch shape: {embeddings.shape}")
```

## 性能指标

基于本地测试的结果：

- **编码速度**: ~48 分子/秒
- **平均时间**: ~21ms/分子
- **嵌入维度**: 128
- **批量处理**: 支持
- **GPU加速**: 支持（如果可用）

## 错误处理

服务包含完整的错误处理机制：

- **无效SMILES**: 返回400错误和详细信息
- **3D构象生成失败**: 使用2D坐标加随机Z坐标
- **几何优化失败**: 继续处理而不优化
- **批处理失败**: 为失败的分子返回零向量

## 部署配置

### Docker Compose

```yaml
schnet:
  build:
    context: ./docker/encoders/schnet
    dockerfile: Dockerfile
  image: molenc-schnet:latest
  container_name: molenc-schnet
  ports:
    - "8004:8000"
  environment:
    - CUDA_VISIBLE_DEVICES=0
    - PYTHONUNBUFFERED=1
  volumes:
    - ./logs:/app/logs
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Nginx路由配置

```nginx
upstream schnet {
    server schnet:8000;
}

location /api/schnet/ {
    proxy_pass http://schnet/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # 超时设置（SchNet 3D计算需要更多时间）
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 120s;
}
```

## 故障排除

### 常见问题

1. **Docker服务启动失败**
   - 检查Docker环境是否正确配置
   - 确认端口8004未被占用
   - 查看日志: `docker logs molenc-schnet`

2. **3D构象生成失败**
   - SchNet会自动回退到2D坐标加随机Z坐标
   - 对于复杂分子，增加`max_conformers`参数

3. **编码速度慢**
   - 启用GPU支持（如果可用）
   - 使用批量处理而非单分子编码
   - 考虑使用流式编码处理大分子

4. **内存使用过高**
   - 减少批处理大小
   - 限制并发请求数
   - 使用较小的模型配置

### 调试工具

```bash
# 查看服务状态
docker compose ps schnet

# 查看实时日志
docker compose logs -f schnet

# 进入容器调试
docker exec -it molenc-schnet bash

# 测试API端点
curl -X GET http://localhost:8004/health
curl -X GET http://localhost:8004/info
```

## 扩展和定制

### 模型定制

可以通过修改`schnet_encoder.py`中的参数来定制模型：

```python
# 创建自定义配置的编码器
encoder = SchNetEncoder(
    hidden_channels=256,      # 更大的嵌入维度
    num_interactions=8,       # 更多的交互层
    cutoff=15.0,              # 更大的截止距离
    num_gaussians=100         # 更多的高斯基函数
)
```

### 添加新端点

在`app.py`中添加新的API端点：

```python
@app.post("/custom_endpoint")
async def custom_endpoint(request: CustomRequest):
    # 实现自定义逻辑
    return {"result": "custom_result"}
```

## 相关资源

- [SchNet论文](https://arxiv.org/abs/1706.08566)
- [RDKit文档](https://www.rdkit.org/docs/)
- [PyTorch文档](https://pytorch.org/docs/)
- [FastAPI文档](https://fastapi.tiangolo.com/)

## 许可证

本项目采用与MolEnc项目相同的许可证。