# MolEnc Docker化分子编码器方案

## 🎯 概述

为MolEnc项目设计完整的Docker化方案，每种分子编码器（Morgan、ChemBERTa、UniMol等）都有独立的Docker容器，预配置好所需环境，支持快速部署和使用。

## 🏗️ 架构设计

### 容器架构
```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Registry                         │
├─────────────────┬─────────────────┬───────────────────────┤
│ morgan-encoder  │ chemberta-encoder│ unimol-encoder      │
│ (CPU/GPU)       │ (CPU/GPU)       │ (CPU/GPU)           │
├─────────────────┴─────────────────┴───────────────────────┤
│              Base Image (molenc-base)                      │
│  ├─ Python 3.9+                                          │
│  ├─ RDKit                                                 │
│  ├─ PyTorch/TensorFlow                                    │
│  ├─ MolEnc Core                                           │
│  └─ Common Utils                                          │
└─────────────────────────────────────────────────────────────┘
```

### 镜像分层策略
1. **基础层**：`molenc-base` - 包含通用依赖
2. **编码器层**：特定编码器依赖和模型
3. **应用层**：API服务和配置

## 📁 项目结构

```
docker/
├── base/
│   ├── Dockerfile.base
│   └── requirements.base.txt
├── encoders/
│   ├── morgan/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app.py
│   ├── chemberta/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── app.py
│   └── unimol/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app.py
├── compose/
│   ├── docker-compose.yml
│   └── docker-compose.gpu.yml
├── scripts/
│   ├── build-all.sh
│   ├── push-all.sh
│   └── run-demo.sh
└── docs/
    ├── quickstart.md
    └── api-reference.md
```

## 🔧 基础镜像配置

### Dockerfile.base
```dockerfile
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装Python依赖
COPY requirements.base.txt .
RUN pip install --no-cache-dir -r requirements.base.txt

# 安装RDKit
RUN pip install rdkit-pypi

# 安装PyTorch (CPU版本)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 复制MolEnc核心代码
COPY molenc/ ./molenc/
COPY setup.py pyproject.toml ./
RUN pip install -e .

# 创建缓存目录
RUN mkdir -p /app/cache /app/models

# 设置环境变量
ENV PYTHONPATH=/app
ENV MOLENC_CACHE_DIR=/app/cache
ENV MOLENC_MODEL_DIR=/app/models

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### requirements.base.txt
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
packaging>=21.0
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
pydantic>=1.8.0
aiofiles>=0.7.0
python-multipart>=0.0.5
```

## 🧬 编码器专用镜像

### Morgan指纹编码器

#### Dockerfile
```dockerfile
FROM molenc-base:latest

# 安装Morgan指纹特定依赖
COPY encoders/morgan/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY encoders/morgan/app.py ./

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 标签
LABEL maintainer="MolEnc Team"
LABEL description="Morgan Fingerprint Encoder"
LABEL version="1.0.0"
```

#### requirements.txt
```
# Morgan指纹依赖 (基础镜像已包含RDKit)
# 无额外依赖
```

#### app.py
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
import json
from molenc.encoders.descriptors.fingerprints.morgan import MorganFingerprintEncoder

app = FastAPI(title="Morgan Fingerprint Encoder", version="1.0.0")

# 全局编码器实例
encoder = None

class EncodeRequest(BaseModel):
    smiles: Union[str, List[str]]
    radius: Optional[int] = 2
    n_bits: Optional[int] = 2048
    use_counts: Optional[bool] = False
    use_features: Optional[bool] = False

class EncodeResponse(BaseModel):
    fingerprints: List[List[float]]
    shape: List[int]
    metadata: dict

@app.on_event("startup")
async def startup_event():
    global encoder
    encoder = MorganFingerprintEncoder()
    print("Morgan encoder initialized")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "encoder": "morgan", "version": "1.0.0"}

@app.get("/info")
async def get_info():
    return {
        "encoder": "morgan",
        "description": "Morgan fingerprint encoder",
        "parameters": {
            "radius": "Fingerprint radius (default: 2)",
            "n_bits": "Number of bits (default: 2048)",
            "use_counts": "Use count-based fingerprints (default: False)",
            "use_features": "Use feature-based fingerprints (default: False)"
        }
    }

@app.post("/encode", response_model=EncodeResponse)
async def encode_molecules(request: EncodeRequest):
    try:
        # 确保smiles是列表
        smiles_list = [request.smiles] if isinstance(request.smiles, str) else request.smiles
        
        # 配置编码器参数
        encoder.radius = request.radius
        encoder.n_bits = request.n_bits
        encoder.use_counts = request.use_counts
        encoder.use_features = request.use_features
        
        # 编码分子
        fingerprints = encoder.encode(smiles_list)
        
        return EncodeResponse(
            fingerprints=fingerprints.tolist(),
            shape=list(fingerprints.shape),
            metadata={
                "n_molecules": len(smiles_list),
                "n_bits": request.n_bits,
                "radius": request.radius,
                "use_counts": request.use_counts,
                "use_features": request.use_features
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/encode/batch")
async def encode_batch(request: EncodeRequest):
    """批量编码接口，支持大文件处理"""
    try:
        smiles_list = [request.smiles] if isinstance(request.smiles, str) else request.smiles
        
        # 分批处理大文件
        batch_size = 1000
        all_fingerprints = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            fps = encoder.encode(batch)
            all_fingerprints.append(fps)
        
        # 合并结果
        fingerprints = np.vstack(all_fingerprints)
        
        return EncodeResponse(
            fingerprints=fingerprints.tolist(),
            shape=list(fingerprints.shape),
            metadata={
                "n_molecules": len(smiles_list),
                "n_bits": request.n_bits,
                "batch_size": batch_size,
                "n_batches": len(range(0, len(smiles_list), batch_size))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### ChemBERTa编码器

#### Dockerfile
```dockerfile
FROM molenc-base:latest

# 安装GPU支持 (如果构建GPU版本)
ARG GPU_VERSION=false
RUN if [ "$GPU_VERSION" = "true" ]; then \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    fi

# 安装ChemBERTa特定依赖
COPY encoders/chemberta/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY encoders/chemberta/app.py ./

# 预下载模型 (可选，减少首次加载时间)
# RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1'); AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')"

HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

LABEL maintainer="MolEnc Team"
LABEL description="ChemBERTa Encoder"
LABEL version="1.0.0"
```

#### requirements.txt
```
transformers>=4.21.0
torch>=1.12.0
tokenizers>=0.13.0
```

#### app.py
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
from molenc.encoders.representations.sequence.chemberta import ChemBERTaEncoder

app = FastAPI(title="ChemBERTa Encoder", version="1.0.0")

encoder = None

class EncodeRequest(BaseModel):
    smiles: Union[str, List[str]]
    model_name: Optional[str] = "seyonec/ChemBERTa-zinc-base-v1"
    pooling_strategy: Optional[str] = "mean"  # "cls", "mean", "max"
    max_length: Optional[int] = 512

class EncodeResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]
    metadata: dict

@app.on_event("startup")
async def startup_event():
    global encoder
    encoder = ChemBERTaEncoder()
    print("ChemBERTa encoder initialized")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "encoder": "chemberta", "version": "1.0.0"}

@app.get("/info")
async def get_info():
    return {
        "encoder": "chemberta",
        "description": "ChemBERTa molecular encoder",
        "parameters": {
            "model_name": "Pre-trained model name",
            "pooling_strategy": "Token pooling strategy (cls/mean/max)",
            "max_length": "Maximum sequence length"
        }
    }

@app.post("/encode", response_model=EncodeResponse)
async def encode_molecules(request: EncodeRequest):
    try:
        smiles_list = [request.smiles] if isinstance(request.smiles, str) else request.smiles
        
        # 配置编码器
        encoder.model_name = request.model_name
        encoder.pooling_strategy = request.pooling_strategy
        encoder.max_length = request.max_length
        
        # 编码分子
        embeddings = encoder.encode(smiles_list)
        
        return EncodeResponse(
            embeddings=embeddings.tolist(),
            shape=list(embeddings.shape),
            metadata={
                "n_molecules": len(smiles_list),
                "embedding_dim": embeddings.shape[1],
                "model_name": request.model_name,
                "pooling_strategy": request.pooling_strategy
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## 🚀 构建和部署

### 构建脚本

#### build-all.sh
```bash
#!/bin/bash
set -e

echo "Building MolEnc Docker images..."

# 构建基础镜像
echo "Building base image..."
docker build -f base/Dockerfile.base -t molenc-base:latest .

# 构建编码器镜像
echo "Building Morgan encoder..."
docker build -f encoders/morgan/Dockerfile -t molenc-morgan:latest .

echo "Building ChemBERTa encoder..."
docker build -f encoders/chemberta/Dockerfile -t molenc-chemberta:latest .

echo "Building GPU versions..."
docker build -f encoders/chemberta/Dockerfile --build-arg GPU_VERSION=true -t molenc-chemberta:gpu .

echo "Build completed!"
```

#### push-all.sh
```bash
#!/bin/bash
set -e

REGISTRY=${REGISTRY:-"your-registry.com"}

echo "Pushing images to registry: $REGISTRY"

docker tag molenc-morgan:latest $REGISTRY/molenc-morgan:latest
docker tag molenc-chemberta:latest $REGISTRY/molenc-chemberta:latest
docker tag molenc-chemberta:gpu $REGISTRY/molenc-chemberta:gpu

docker push $REGISTRY/molenc-morgan:latest
docker push $REGISTRY/molenc-chemberta:latest
docker push $REGISTRY/molenc-chemberta:gpu

echo "Push completed!"
```

### Docker Compose配置

#### docker-compose.yml
```yaml
version: '3.8'

services:
  morgan:
    image: molenc-morgan:latest
    container_name: molenc-morgan
    ports:
      - "8001:8000"
    environment:
      - MOLENC_CACHE_DIR=/app/cache
      - PYTHONUNBUFFERED=1
    volumes:
      - ./cache:/app/cache
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  chemberta:
    image: molenc-chemberta:latest
    container_name: molenc-chemberta
    ports:
      - "8002:8000"
    environment:
      - MOLENC_CACHE_DIR=/app/cache
      - MOLENC_MODEL_DIR=/app/models
      - PYTHONUNBUFFERED=1
    volumes:
      - ./cache:/app/cache
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: molenc-gateway
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - morgan
      - chemberta
    restart: unless-stopped

volumes:
  cache:
  models:
```

#### docker-compose.gpu.yml
```yaml
version: '3.8'

services:
  chemberta-gpu:
    image: molenc-chemberta:gpu
    container_name: molenc-chemberta-gpu
    ports:
      - "8003:8000"
    environment:
      - MOLENC_CACHE_DIR=/app/cache
      - MOLENC_MODEL_DIR=/app/models
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
    volumes:
      - ./cache:/app/cache
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

## 🔧 Nginx网关配置

#### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream morgan {
        server morgan:8000;
    }
    
    upstream chemberta {
        server chemberta:8000;
    }
    
    server {
        listen 80;
        
        # Morgan指纹编码器
        location /api/morgan/ {
            proxy_pass http://morgan/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # ChemBERTa编码器
        location /api/chemberta/ {
            proxy_pass http://chemberta/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 健康检查
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

## 📖 使用示例

### 快速启动
```bash
# 1. 克隆项目
git clone https://github.com/your-org/molenc-docker.git
cd molenc-docker

# 2. 启动服务
docker-compose up -d

# 3. 检查状态
curl http://localhost/health

# 4. 使用Morgan编码器
curl -X POST http://localhost/api/morgan/encode \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "n_bits": 2048}'

# 5. 使用ChemBERTa编码器  
curl -X POST http://localhost/api/chemberta/encode \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "model_name": "seyonec/ChemBERTa-zinc-base-v1"}'
```

### Python客户端示例
```python
import requests
import json

# Morgan指纹编码
response = requests.post(
    "http://localhost/api/morgan/encode",
    json={"smiles": ["CCO", "CCCO"], "n_bits": 1024}
)
fingerprints = response.json()["fingerprints"]
print(f"Morgan fingerprints shape: {len(fingerprints)}x{len(fingerprints[0])}")

# ChemBERTa编码
response = requests.post(
    "http://localhost/api/chemberta/encode", 
    json={"smiles": ["CCO", "CCCO"], "pooling_strategy": "mean"}
)
embeddings = response.json()["embeddings"]
print(f"ChemBERTa embeddings shape: {len(embeddings)}x{len(embeddings[0])}")
```

### 批量处理
```python
import pandas as pd

# 读取分子数据
df = pd.read_csv("molecules.csv")
smiles_list = df["smiles"].tolist()

# 分批处理
batch_size = 100
all_embeddings = []

for i in range(0, len(smiles_list), batch_size):
    batch = smiles_list[i:i+batch_size]
    response = requests.post(
        "http://localhost/api/chemberta/encode/batch",
        json={"smiles": batch, "batch_size": batch_size}
    )
    all_embeddings.extend(response.json()["embeddings"])

# 保存结果
df["embedding"] = all_embeddings
df.to_csv("molecules_with_embeddings.csv", index=False)
```

## 🚀 GPU支持

### 构建GPU镜像
```bash
# 构建GPU版本
docker build -f encoders/chemberta/Dockerfile \
  --build-arg GPU_VERSION=true \
  -t molenc-chemberta:gpu .

# 运行GPU容器
docker run --gpus all -p 8003:8000 molenc-chemberta:gpu
```

### 使用GPU Compose
```bash
# 启动GPU服务
docker-compose -f docker-compose.gpu.yml up -d

# 验证GPU可用性
curl http://localhost:8003/info
```

## 📊 性能优化

### 1. 模型预加载
在Dockerfile中预下载常用模型：
```dockerfile
# 预下载ChemBERTa模型
RUN python -c "
from transformers import AutoTokenizer, AutoModel;
tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1');
model = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
"
```

### 2. 缓存优化
```yaml
# 在compose中配置缓存
volumes:
  - type: bind
    source: ./cache
    target: /app/cache
    bind:
      propagation: rprivate
```

### 3. 批量处理优化
```python
# 在app.py中实现智能批处理
@app.post("/encode/optimized")
async def encode_optimized(request: EncodeRequest):
    # 根据分子数量自动选择批大小
    n_molecules = len(request.smiles)
    optimal_batch_size = min(1000, max(32, n_molecules // 10))
    
    # 并行处理多个批次
    # ... 实现细节
```

## 🔍 监控和日志

### 健康检查
每个容器提供健康检查端点：
```bash
# 检查服务状态
curl http://localhost:8001/health  # Morgan
curl http://localhost:8002/health  # ChemBERTa
```

### 日志查看
```bash
# 查看容器日志
docker logs molenc-morgan
docker logs molenc-chemberta

# 实时监控
docker logs -f molenc-morgan
```

### 性能监控
```python
# 在app.py中添加性能监控
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

## 🛠️ 故障排除

### 常见问题

1. **容器启动失败**
```bash
# 检查日志
docker logs molenc-morgan

# 检查端口冲突
netstat -tulpn | grep :8001
```

2. **模型下载失败**
```bash
# 检查网络连接
docker exec molenc-chemberta curl -I https://huggingface.co

# 手动下载模型
docker exec -it molenc-chemberta python -c "
from transformers import AutoTokenizer;
AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
"
```

3. **内存不足**
```bash
# 检查内存使用
docker stats

# 调整批大小
# 修改app.py中的batch_size参数
```

## 🔄 更新和维护

### 更新镜像
```bash
# 拉取最新代码
git pull origin main

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose down && docker-compose up -d
```

### 备份和恢复
```bash
# 备份缓存
tar -czf molenc_cache_backup.tar.gz ./cache/

# 恢复缓存
tar -xzf molenc_cache_backup.tar.gz
```

## 📚 扩展开发

### 添加新编码器
1. 在`encoders/`目录创建新文件夹
2. 编写Dockerfile和requirements.txt
3. 实现app.py（遵循统一API规范）
4. 更新docker-compose.yml
5. 添加到构建脚本

### API标准化
所有编码器必须实现以下端点：
- `GET /health` - 健康检查
- `GET /info` - 编码器信息
- `POST /encode` - 单分子编码
- `POST /encode/batch` - 批量编码

这个方案提供了完整的Docker化分子编码器解决方案，支持快速部署、易于扩展和高效使用。