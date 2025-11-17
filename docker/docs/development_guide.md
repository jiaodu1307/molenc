# MolEnc 开发指南

本指南详细介绍了MolEnc Docker项目的开发流程、架构设计和扩展方法。

## 项目架构

### 总体架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Gateway │    │   Morgan        │    │   ChemBERTa     │    │   MACCS         │
│   (Port 80/8080)│◄──►│   Encoder       │◄──►│   Encoder       │◄──►│   Encoder       │
│   - 负载均衡    │    │   (Port 8001)   │    │   (Port 8002)   │    │   (Port 8003)   │
│   - 路由转发    │    │   - RDKit       │    │   - Transformers│    │   - RDKit       │
│   - 健康检查    │    │   - Morgan FP   │    │   - PyTorch     │    │   - MACCS Keys  │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 目录结构
```
molenc/
├── docker/
│   ├── base/                    # 基础镜像构建
│   │   ├── Dockerfile.base      # 基础Dockerfile
│   │   └── requirements.base.txt # 基础依赖
│   ├── encoders/                # 编码器容器
│   │   ├── morgan/              # Morgan编码器
│   │   │   ├── Dockerfile       # Morgan Dockerfile
│   │   │   ├── requirements.txt # Morgan依赖
│   │   │   └── app.py           # Morgan API服务
│   │   ├── chemberta/           # ChemBERTa编码器
│   │   │   ├── Dockerfile       # ChemBERTa Dockerfile
│   │   │   ├── requirements.txt # ChemBERTa依赖
│   │   │   └── app.py           # ChemBERTa API服务
│   │   └── maccs/               # MACCS编码器
│   │       ├── Dockerfile       # MACCS Dockerfile
│   │       ├── requirements.txt # MACCS依赖
│   │       ├── app.py           # MACCS API服务
│   │       └── maccs_encoder.py # MACCS编码器实现
│   ├── compose/                 # Docker Compose配置
│   │   └── docker-compose.yml   # 服务编排
│   ├── scripts/                 # 脚本工具
│   │   ├── quickstart.sh        # 快速启动脚本
│   │   └── build.sh             # 构建脚本
│   ├── examples/                # 示例代码
│   │   ├── client_example.py    # Python客户端
│   │   └── api_test.py          # API测试工具
│   ├── docs/                    # 文档
│   │   ├── deployment_guide.md  # 部署指南
│   │   ├── api_reference.md     # API参考
│   │   └── development_guide.md # 开发指南
│   └── nginx.conf               # Nginx配置
└── shared/                      # 共享代码
    ├── morgan_encoder.py        # Morgan编码器实现
    └── chemberta_encoder.py     # ChemBERTa编码器实现
```

## 开发环境设置

### 1. 基础环境
```bash
# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 安装Python依赖
pip install rdkit transformers torch fastapi uvicorn requests
```

### 2. 开发工具
```bash
# 安装开发工具
pip install pytest black flake8 mypy pre-commit

# 配置Git钩子
pre-commit install
```

### 3. 项目初始化
```bash
# 克隆项目
git clone <repository-url>
cd molenc

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r docker/base/requirements.base.txt
```

## 编码器开发

### 添加新编码器

#### 1. 创建编码器目录
```bash
mkdir -p docker/encoders/newencoder
cd docker/encoders/newencoder
```

#### 2. 创建Dockerfile
```dockerfile
# docker/encoders/newencoder/Dockerfile
FROM molenc-base:latest

# 安装特定依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# 复制应用代码
COPY app.py /app/
COPY newencoder_encoder.py /app/

# 设置环境变量
ENV ENCODER_TYPE=newencoder
ENV DEFAULT_PORT=8000

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 3. 创建依赖文件
```txt
# docker/encoders/newencoder/requirements.txt
# 添加编码器特定的依赖
numpy>=1.21.0
scikit-learn>=1.0.0
# 其他依赖...
```

#### 4. 创建编码器实现
```python
# docker/encoders/newencoder/newencoder_encoder.py
from typing import List, Dict, Any, Optional
import numpy as np

class NewEncoder:
    """新编码器实现"""
    
    def __init__(self, param1: str = "default", param2: int = 100):
        self.param1 = param1
        self.param2 = param2
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        # 模型初始化逻辑
        pass
    
    def encode(self, smiles: str, **kwargs) -> List[float]:
        """编码单个SMILES"""
        # 编码逻辑
        return [0.0] * self.param2
    
    def encode_batch(self, smiles_list: List[str], **kwargs) -> List[List[float]]:
        """批量编码SMILES"""
        return [self.encode(smiles, **kwargs) for smiles in smiles_list]
    
    def get_info(self) -> Dict[str, Any]:
        """获取编码器信息"""
        return {
            "name": "newencoder",
            "description": "新编码器描述",
            "parameters": {
                "param1": {"type": "string", "default": "default"},
                "param2": {"type": "integer", "default": 100}
            }
        }
```

#### 5. 创建API服务
```python
# docker/encoders/newencoder/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from newencoder_encoder import NewEncoder

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="NewEncoder API",
    description="新编码器API服务",
    version="1.0.0"
)

# 请求模型
class EncodeRequest(BaseModel):
    smiles: List[str] = Field(..., description="SMILES字符串列表")
    param1: Optional[str] = Field("default", description="参数1")
    param2: Optional[int] = Field(100, description="参数2", ge=1, le=1000)

# 响应模型
class EncodeResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

# 初始化编码器
encoder = NewEncoder()

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("NewEncoder服务启动")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "success": True,
        "message": "NewEncoder编码器运行正常",
        "data": {"status": "healthy", "encoder": "newencoder"},
        "metadata": {"timestamp": "2024-01-15T10:30:00Z"}
    }

@app.get("/info")
async def get_info():
    """获取编码器信息"""
    return {
        "success": True,
        "message": "NewEncoder编码器信息",
        "data": encoder.get_info(),
        "metadata": {"timestamp": "2024-01-15T10:30:00Z"}
    }

@app.post("/encode", response_model=EncodeResponse)
async def encode(request: EncodeRequest):
    """编码SMILES"""
    try:
        # 重新初始化编码器（如果需要）
        if request.param1 != encoder.param1 or request.param2 != encoder.param2:
            encoder.__init__(param1=request.param1, param2=request.param2)
        
        # 批量编码
        results = encoder.encode_batch(request.smiles)
        
        return {
            "success": True,
            "message": "编码成功",
            "data": {
                "encodings": results,
                "shape": [len(request.smiles), request.param2]
            },
            "metadata": {
                "encoder": "newencoder",
                "param1": request.param1,
                "param2": request.param2,
                "n_molecules": len(request.smiles)
            }
        }
    except Exception as e:
        logger.error(f"编码失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"编码失败: {str(e)}")

@app.post("/encode_batch", response_model=EncodeResponse)
async def encode_batch(request: EncodeRequest):
    """批量编码SMILES"""
    return await encode(request)
```

#### 6. 更新Docker Compose
```yaml
# 在docker-compose.yml中添加新服务
services:
  newencoder:
    build:
      context: ../../
      dockerfile: docker/encoders/newencoder/Dockerfile
    image: molenc-newencoder:latest
    container_name: molenc-newencoder
    ports:
      - "8003:8000"
    environment:
      - ENCODER_TYPE=newencoder
    volumes:
      - ./cache:/app/cache
      - ./models:/app/models
      - ./logs:/app/logs
    networks:
      - molenc-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

#### 7. 更新Nginx配置
```nginx
# 在nginx.conf中添加新上游
upstream newencoder_backend {
    server newencoder:8000;
}

# 添加新位置
location /api/newencoder/ {
    proxy_pass http://newencoder_backend/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_connect_timeout 30s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
}
```

### 编码器最佳实践

#### 1. 错误处理
```python
def encode_smiles(self, smiles: str, **kwargs) -> List[float]:
    """编码SMILES字符串"""
    try:
        # 验证SMILES
        if not self._is_valid_smiles(smiles):
            raise ValueError(f"无效的SMILES: {smiles}")
        
        # 编码逻辑
        encoding = self._perform_encoding(smiles, **kwargs)
        
        # 验证输出
        if not encoding or len(encoding) == 0:
            raise RuntimeError(f"编码结果为空: {smiles}")
        
        return encoding
        
    except ValueError as e:
        # 重新抛出参数错误
        raise e
    except Exception as e:
        # 记录错误并抛出
        logger.error(f"编码失败 {smiles}: {str(e)}")
        raise RuntimeError(f"编码失败: {str(e)}")
```

#### 2. 性能优化
```python
import time
from functools import lru_cache

class OptimizedEncoder:
    def __init__(self):
        self.cache = {}
        self.batch_size = 32
    
    @lru_cache(maxsize=1000)
    def _get_model(self, param1: str, param2: int):
        """缓存模型实例"""
        return self._create_model(param1, param2)
    
    def encode_batch(self, smiles_list: List[str], **kwargs) -> List[List[float]]:
        """优化的批量编码"""
        results = []
        
        # 分批处理
        for i in range(0, len(smiles_list), self.batch_size):
            batch = smiles_list[i:i + self.batch_size]
            
            # 使用缓存
            cache_key = tuple(batch)
            if cache_key in self.cache:
                results.extend(self.cache[cache_key])
                continue
            
            # 批量处理
            batch_results = self._process_batch(batch, **kwargs)
            
            # 缓存结果
            self.cache[cache_key] = batch_results
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch: List[str], **kwargs) -> List[List[float]]:
        """处理一批SMILES"""
        # 向量化处理
        start_time = time.time()
        
        # 批量编码逻辑
        results = []
        for smiles in batch:
            result = self.encode(smiles, **kwargs)
            results.append(result)
        
        processing_time = time.time() - start_time
        logger.info(f"批量处理 {len(batch)} 个分子，耗时 {processing_time:.3f}s")
        
        return results
```

#### 3. 内存管理
```python
import gc
import torch

class MemoryEfficientEncoder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
    
    def _clear_memory(self):
        """清理内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def encode_batch(self, smiles_list: List[str], **kwargs) -> List[List[float]]:
        """内存高效的批量编码"""
        try:
            # 分批处理避免内存溢出
            batch_size = min(kwargs.get("batch_size", 32), len(smiles_list))
            results = []
            
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i:i + batch_size]
                
                # 处理批次
                batch_results = self._process_batch_gpu(batch, **kwargs)
                results.extend(batch_results)
                
                # 定期清理内存
                if i % (batch_size * 10) == 0:
                    self._clear_memory()
            
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU内存不足，切换到CPU处理")
                self.device = torch.device("cpu")
                return self.encode_batch(smiles_list, **kwargs)
            else:
                raise e
```

## API开发

### FastAPI最佳实践

#### 1. 请求验证
```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class EncodeRequest(BaseModel):
    smiles: List[str] = Field(..., min_items=1, max_items=1000, description="SMILES字符串列表")
    n_bits: Optional[int] = Field(1024, ge=64, le=16384, description="指纹位数")
    radius: Optional[int] = Field(2, ge=1, le=5, description="指纹半径")
    use_features: Optional[bool] = Field(False, description="是否使用特征")
    
    @validator('smiles')
    def validate_smiles(cls, v):
        """验证SMILES格式"""
        if not all(isinstance(s, str) and len(s.strip()) > 0 for s in v):
            raise ValueError('SMILES必须是有效的字符串')
        return [s.strip() for s in v]
    
    @validator('n_bits')
    def validate_n_bits(cls, v):
        """验证指纹位数"""
        if v & (v - 1) != 0:  # 检查是否为2的幂
            raise ValueError('n_bits必须是2的幂 (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "smiles": ["CCO", "c1ccccc1", "CCCO"],
                "n_bits": 1024,
                "radius": 2,
                "use_features": false
            }
        }
```

#### 2. 响应标准化
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict

class StandardResponse(JSONResponse):
    """标准化响应"""
    def __init__(self, success: bool, message: str, data: Any = None, metadata: Dict = None):
        content = {
            "success": success,
            "message": message,
            "data": data or {},
            "metadata": metadata or {}
        }
        super().__init__(content=content)

def create_success_response(data: Any = None, message: str = "操作成功", metadata: Dict = None):
    """创建成功响应"""
    return StandardResponse(True, message, data, metadata)

def create_error_response(message: str, status_code: int = 500, details: Dict = None):
    """创建错误响应"""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "error": details or {},
            "metadata": {"timestamp": datetime.now().isoformat()}
        }
    )

# 使用示例
@app.post("/encode")
async def encode(request: EncodeRequest):
    try:
        result = encoder.encode_batch(request.smiles, **request.dict())
        return create_success_response(
            data={"fingerprints": result},
            message="编码成功",
            metadata={"n_molecules": len(request.smiles)}
        )
    except ValueError as e:
        return create_error_response(str(e), status_code=400)
    except Exception as e:
        logger.error(f"编码失败: {str(e)}")
        return create_error_response("服务器内部错误", status_code=500)
```

#### 3. 异常处理
```python
from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """处理值错误"""
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": "参数错误",
            "error": {
                "type": "ValueError",
                "details": str(exc)
            },
            "metadata": {"timestamp": datetime.now().isoformat()}
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理通用异常"""
    logger.error(f"未处理的异常: {str(exc)}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误",
            "error": {
                "type": type(exc).__name__,
                "details": "服务器遇到错误，请稍后重试"
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "trace_id": "error_12345"  # 用于错误追踪
            }
        }
    )
```

## 测试开发

### 单元测试
```python
# tests/test_morgan_encoder.py
import pytest
from morgan_encoder import MorganEncoder

class TestMorganEncoder:
    
    @pytest.fixture
    def encoder(self):
        return MorganEncoder()
    
    def test_encode_valid_smiles(self, encoder):
        """测试有效SMILES编码"""
        smiles = "CCO"
        result = encoder.encode(smiles)
        
        assert isinstance(result, list)
        assert len(result) == 1024  # 默认1024位
        assert all(isinstance(x, (int, float)) for x in result)
    
    def test_encode_invalid_smiles(self, encoder):
        """测试无效SMILES编码"""
        with pytest.raises(ValueError):
            encoder.encode("invalid_smiles")
    
    def test_encode_batch(self, encoder):
        """测试批量编码"""
        smiles_list = ["CCO", "c1ccccc1", "CCCO"]
        results = encoder.encode_batch(smiles_list)
        
        assert len(results) == 3
        assert all(len(result) == 1024 for result in results)
    
    def test_parameter_validation(self, encoder):
        """测试参数验证"""
        with pytest.raises(ValueError):
            encoder.encode("CCO", n_bits=32)  # n_bits太小
        
        with pytest.raises(ValueError):
            encoder.encode("CCO", radius=0)  # radius太小
    
    def test_performance(self, encoder):
        """测试性能"""
        import time
        
        smiles_list = ["CCO"] * 100
        start_time = time.time()
        results = encoder.encode_batch(smiles_list)
        end_time = time.time()
        
        assert len(results) == 100
        assert end_time - start_time < 5.0  # 100个分子应在5秒内完成
```

### API测试
```python
# tests/test_api.py
import pytest
import requests
import json

class TestMolEncAPI:
    
    @pytest.fixture
    def base_url(self):
        return "http://localhost"
    
    def test_health_check(self, base_url):
        """测试健康检查"""
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "healthy" in data["message"]
    
    def test_morgan_encode(self, base_url):
        """测试Morgan编码"""
        payload = {
            "smiles": ["CCO", "c1ccccc1"],
            "n_bits": 1024,
            "radius": 2
        }
        
        response = requests.post(f"{base_url}/api/morgan/encode", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert len(data["data"]["fingerprints"]) == 2
        assert len(data["data"]["fingerprints"][0]) == 1024
    
    def test_chemberta_encode(self, base_url):
        """测试ChemBERTa编码"""
        payload = {
            "smiles": ["CCO", "c1ccccc1"],
            "model_name": "seyonec/ChemBERTa-zinc-base-v1",
            "pooling_strategy": "mean"
        }
        
        response = requests.post(f"{base_url}/api/chemberta/encode", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert len(data["data"]["embeddings"]) == 2
        assert len(data["data"]["embeddings"][0]) == 768  # ChemBERTa默认768维
    
    def test_error_handling(self, base_url):
        """测试错误处理"""
        # 无效SMILES
        payload = {"smiles": ["invalid_smiles"]}
        response = requests.post(f"{base_url}/api/morgan/encode", json=payload)
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] == False
        assert "无效" in data["message"]
    
    def test_parameter_validation(self, base_url):
        """测试参数验证"""
        # 无效的n_bits
        payload = {
            "smiles": ["CCO"],
            "n_bits": 100  # 不是2的幂
        }
        
        response = requests.post(f"{base_url}/api/morgan/encode", json=payload)
        assert response.status_code == 422  # 验证错误
```

### 性能测试
```python
# tests/test_performance.py
import pytest
import requests
import time
import concurrent.futures

class TestPerformance:
    
    @pytest.fixture
    def base_url(self):
        return "http://localhost"
    
    def test_single_molecule_performance(self, base_url):
        """测试单分子编码性能"""
        payload = {"smiles": ["CCO"]}
        
        # 预热
        for _ in range(3):
            requests.post(f"{base_url}/api/morgan/encode", json=payload)
        
        # 性能测试
        times = []
        for _ in range(100):
            start_time = time.time()
            response = requests.post(f"{base_url}/api/morgan/encode", json=payload)
            end_time = time.time()
            
            assert response.status_code == 200
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        assert avg_time < 0.1  # 平均响应时间应小于100ms
    
    def test_batch_performance(self, base_url):
        """测试批量编码性能"""
        # 测试不同批量大小的性能
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            smiles_list = ["CCO"] * batch_size
            payload = {"smiles": smiles_list}
            
            start_time = time.time()
            response = requests.post(f"{base_url}/api/morgan/encode", json=payload)
            end_time = time.time()
            
            assert response.status_code == 200
            
            total_time = end_time - start_time
            time_per_molecule = total_time / batch_size
            
            print(f"批量大小 {batch_size}: 总时间 {total_time:.3f}s, 单分子 {time_per_molecule:.3f}s")
            
            # 批量处理应该有更好的性能
            if batch_size > 10:
                assert time_per_molecule < 0.05  # 批量处理单分子时间应小于50ms
    
    def test_concurrent_performance(self, base_url):
        """测试并发性能"""
        def make_request():
            payload = {"smiles": ["CCO", "c1ccccc1"]}
            start_time = time.time()
            response = requests.post(f"{base_url}/api/morgan/encode", json=payload)
            end_time = time.time()
            
            return response.status_code == 200, end_time - start_time
        
        # 并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_count = sum(1 for success, _ in results if success)
        avg_time = sum(time for _, time in results) / len(results)
        
        assert success_count == 50  # 所有请求都应成功
        assert avg_time < 0.2  # 并发平均响应时间应小于200ms
```

## 部署和运维

### Docker优化

#### 1. 多阶段构建
```dockerfile
# docker/encoders/chemberta/Dockerfile
# 第一阶段：构建阶段
FROM python:3.9-slim as builder

WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --user --no-cache-dir -r requirements.txt

# 第二阶段：运行阶段
FROM python:3.9-slim

WORKDIR /app

# 复制已安装的依赖
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY app.py .
COPY chemberta_encoder.py .

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. 镜像优化
```bash
# 使用.dockerignore减少构建上下文
# .dockerignore
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git/
.gitignore
README.md
Dockerfile
docker-compose.yml
.dockerignore

# 构建优化镜像
docker build -t molenc-optimized:latest \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from molenc-base:latest \
  -f docker/encoders/chemberta/Dockerfile .

# 扫描镜像安全性
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/app aquasec/trivy:latest image molenc-optimized:latest
```

### 监控和日志

#### 1. 结构化日志
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_request(self, method: str, path: str, status_code: int, duration: float, **kwargs):
        """记录请求日志"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "info",
            "type": "request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration * 1000,
            **kwargs
        }
        self.logger.info(json.dumps(log_data))
    
    def log_error(self, error_type: str, message: str, **kwargs):
        """记录错误日志"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "error",
            "type": "error",
            "error_type": error_type,
            "message": message,
            **kwargs
        }
        self.logger.error(json.dumps(log_data))
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """记录性能日志"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "info",
            "type": "performance",
            "operation": operation,
            "duration_ms": duration * 1000,
            **kwargs
        }
        self.logger.info(json.dumps(log_data))

# 使用示例
logger = StructuredLogger("morgan_encoder")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """请求日志中间件"""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.log_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration,
        client_ip=request.client.host if request.client else None
    )
    
    return response
```

#### 2. 性能监控
```python
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Prometheus指标
REQUEST_COUNT = Counter('molenc_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('molenc_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('molenc_active_connections', 'Active connections')
MEMORY_USAGE = Gauge('molenc_memory_usage_bytes', 'Memory usage')
CPU_USAGE = Gauge('molenc_cpu_usage_percent', 'CPU usage')

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
    
    def collect_system_metrics(self):
        """收集系统指标"""
        # 内存使用
        memory = psutil.virtual_memory()
        MEMORY_USAGE.set(memory.used)
        
        # CPU使用
        cpu_percent = psutil.cpu_percent(interval=1)
        CPU_USAGE.set(cpu_percent)
        
        # 活跃连接数
        connections = len(psutil.net_connections())
        ACTIVE_CONNECTIONS.set(connections)
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """记录请求指标"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    collector = MetricsCollector()
    collector.collect_system_metrics()
    
    return Response(content=generate_latest(), media_type="text/plain")
```

### CI/CD集成

#### 1. GitHub Actions
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r docker/base/requirements.base.txt
        pip install pytest pytest-cov flake8 black mypy
    
    - name: Lint with flake8
      run: |
        flake8 docker/encoders/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 docker/encoders/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Format check with black
      run: black --check docker/encoders/
    
    - name: Type check with mypy
      run: mypy docker/encoders/ --ignore-missing-imports
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=docker/encoders/ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push base image
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./docker/base/Dockerfile.base
        push: true
        tags: molenc-base:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Build and push encoder images
      run: |
        docker build -t molenc-morgan:latest -f docker/encoders/morgan/Dockerfile .
        docker build -t molenc-chemberta:latest -f docker/encoders/chemberta/Dockerfile .
        docker push molenc-morgan:latest
        docker push molenc-chemberta:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        # 部署到测试环境
        ssh user@staging-server 'cd /opt/molenc && ./deploy.sh'
    
    - name: Run integration tests
      run: |
        # 运行集成测试
        python docker/examples/api_test.py --url http://staging-server
    
    - name: Deploy to production
      if: success()
      run: |
        # 部署到生产环境
        ssh user@prod-server 'cd /opt/molenc && ./deploy.sh'
```

## 扩展和定制

### 添加新的分子特征

#### 1. 分子描述符编码器
```python
# docker/encoders/descriptors/descriptors_encoder.py
from typing import List, Dict, Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

class MolecularDescriptorsEncoder:
    """分子描述符编码器"""
    
    def __init__(self, descriptor_types: List[str] = None):
        self.descriptor_types = descriptor_types or [
            "MolWt", "LogP", "NumHDonors", "NumHAcceptors", 
            "TPSA", "NumRotatableBonds", "NumAromaticRings"
        ]
        self.descriptors = self._get_descriptor_functions()
    
    def _get_descriptor_functions(self) -> Dict[str, callable]:
        """获取描述符函数"""
        available_descriptors = {
            "MolWt": Descriptors.MolWt,
            "LogP": Descriptors.MolLogP,
            "NumHDonors": Descriptors.NumHDonors,
            "NumHAcceptors": Descriptors.NumHAcceptors,
            "TPSA": Descriptors.TPSA,
            "NumRotatableBonds": Descriptors.NumRotatableBonds,
            "NumAromaticRings": Descriptors.NumAromaticRings,
            "NumAtoms": lambda mol: mol.GetNumAtoms(),
            "NumBonds": lambda mol: mol.GetNumBonds(),
            "FractionCSP3": Descriptors.FractionCSP3,
            "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
            "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
            "NumHeteroatoms": Descriptors.NumHeteroatoms,
            "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
        }
        
        return {k: v for k, v in available_descriptors.items() if k in self.descriptor_types}
    
    def encode(self, smiles: str) -> List[float]:
        """编码单个SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效的SMILES: {smiles}")
        
        descriptors = []
        for desc_name, desc_func in self.descriptors.items():
            try:
                value = desc_func(mol)
                descriptors.append(float(value))
            except Exception as e:
                logger.warning(f"计算描述符 {desc_name} 失败: {str(e)}")
                descriptors.append(0.0)
        
        return descriptors
    
    def encode_batch(self, smiles_list: List[str]) -> List[List[float]]:
        """批量编码SMILES"""
        return [self.encode(smiles) for smiles in smiles_list]
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        return list(self.descriptors.keys())
    
    def get_info(self) -> Dict[str, Any]:
        """获取编码器信息"""
        return {
            "name": "molecular_descriptors",
            "description": "分子描述符编码器",
            "n_features": len(self.descriptor_types),
            "feature_names": self.get_feature_names(),
            "parameters": {
                "descriptor_types": {
                    "type": "list[string]",
                    "default": self.descriptor_types,
                    "available_descriptors": list(self._get_descriptor_functions().keys())
                }
            }
        }
```

#### 2. 图神经网络编码器
```python
# docker/encoders/gnn/gnn_encoder.py
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
from typing import List, Dict, Any

class GNNMoleculeEncoder(nn.Module):
    """图神经网络分子编码器"""
    
    def __init__(self, input_dim: int = 74, hidden_dim: int = 128, output_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
    
    def forward(self, x, edge_index, batch):
        """前向传播"""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, p=0.1, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        return x

class GNNEncoder:
    """GNN分子编码器包装类"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = GNNMoleculeEncoder()
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
    
    def smiles_to_graph(self, smiles: str) -> Data:
        """将SMILES转换为图数据"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效的SMILES: {smiles}")
        
        # 获取原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = self._get_atom_features(atom)
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # 获取边信息
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.extend([[i, j], [j, i]])  # 无向图
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def _get_atom_features(self, atom) -> List[float]:
        """获取原子特征"""
        features = []
        
        # 基本特征
        features.append(atom.GetAtomicNum())
        features.append(atom.GetDegree())
        features.append(atom.GetFormalCharge())
        features.append(int(atom.GetHybridization()))
        features.append(int(atom.GetIsAromatic()))
        features.append(atom.GetNumRadicalElectrons())
        features.append(atom.GetTotalNumHs())
        
        # 周期性特征
        features.extend(self._one_hot(atom.GetAtomicNum() - 1, list(range(100))))
        features.extend(self._one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]))
        features.extend(self._one_hot(int(atom.GetHybridization()), list(range(7))))
        
        return features
    
    def _one_hot(self, value: int, choices: List[int]) -> List[int]:
        """独热编码"""
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding
    
    def encode(self, smiles: str) -> List[float]:
        """编码单个SMILES"""
        graph_data = self.smiles_to_graph(smiles)
        
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            embedding = self.model(graph_data.x, graph_data.edge_index, torch.zeros(graph_data.x.size(0), dtype=torch.long).to(self.device))
        
        return embedding.cpu().numpy().tolist()[0]
    
    def encode_batch(self, smiles_list: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量编码SMILES"""
        results = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            batch_graphs = [self.smiles_to_graph(smiles) for smiles in batch_smiles]
            
            # 创建批次
            batch = torch_geometric.data.Batch.from_data_list(batch_graphs)
            batch = batch.to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(batch.x, batch.edge_index, batch.batch)
            
            results.extend(embeddings.cpu().numpy().tolist())
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """获取编码器信息"""
        return {
            "name": "gnn_encoder",
            "description": "图神经网络分子编码器",
            "model_type": "GCN",
            "output_dim": self.model.convs[-1].out_channels,
            "num_layers": self.model.num_layers,
            "device": str(self.device)
        }
```

### 自定义API中间件

#### 1. 认证中间件
```python
# docker/middleware/auth_middleware.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import time
from typing import Optional

class AuthMiddleware:
    """认证中间件"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.security = HTTPBearer()
    
    def create_token(self, user_id: str, expires_in: int = 3600) -> str:
        """创建JWT令牌"""
        payload = {
            "user_id": user_id,
            "exp": time.time() + expires_in,
            "iat": time.time()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[dict]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def __call__(self, request: Request):
        """中间件调用"""
        # 跳过健康检查和文档端点
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return
        
        # 获取认证头
        authorization = request.headers.get("Authorization")
        if not authorization:
            raise HTTPException(status_code=401, detail="缺少认证头")
        
        # 验证令牌
        try:
            scheme, token = authorization.split(" ", 1)
            if scheme.lower() != "bearer":
                raise HTTPException(status_code=401, detail="无效的认证方案")
            
            payload = self.verify_token(token)
            if not payload:
                raise HTTPException(status_code=401, detail="无效或过期的令牌")
            
            # 将用户信息添加到请求状态
            request.state.user = payload
            
        except ValueError:
            raise HTTPException(status_code=401, detail="无效的认证头格式")

# 使用示例
from fastapi import FastAPI, Depends

app = FastAPI()
auth_middleware = AuthMiddleware(secret_key="your-secret-key")

@app.middleware("http")
async def auth_middleware_func(request: Request, call_next):
    try:
        await auth_middleware(request)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    
    response = await call_next(request)
    return response
```

#### 2. 限流中间件
```python
# docker/middleware/rate_limit_middleware.py
import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import Request, HTTPException

class RateLimiter:
    """限流器"""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, int]]:
        """检查是否允许请求"""
        now = time.time()
        
        # 清理过期的请求记录
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < 3600  # 保留1小时内的记录
        ]
        
        # 检查分钟限制
        recent_requests = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < 60
        ]
        
        if len(recent_requests) >= self.requests_per_minute:
            return False, {
                "allowed": False,
                "limit": self.requests_per_minute,
                "window": "1 minute",
                "remaining": 0,
                "retry_after": int(60 - (now - min(recent_requests)))
            }
        
        # 检查小时限制
        if len(self.requests[client_id]) >= self.requests_per_hour:
            return False, {
                "allowed": False,
                "limit": self.requests_per_hour,
                "window": "1 hour",
                "remaining": 0,
                "retry_after": int(3600 - (now - min(self.requests[client_id])))
            }
        
        # 记录当前请求
        self.requests[client_id].append(now)
        
        # 计算剩余配额
        minute_remaining = self.requests_per_minute - len(recent_requests) - 1
        hour_remaining = self.requests_per_hour - len(self.requests[client_id])
        
        return True, {
            "allowed": True,
            "minute_remaining": minute_remaining,
            "hour_remaining": hour_remaining
        }

class RateLimitMiddleware:
    """限流中间件"""
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.rate_limiter = RateLimiter(requests_per_minute, requests_per_hour)
    
    async def __call__(self, request: Request):
        """中间件调用"""
        # 获取客户端ID（可以是IP地址或用户ID）
        client_id = request.client.host if request.client else "unknown"
        
        # 检查限流
        allowed, info = self.rate_limiter.is_allowed(client_id)
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，请 {info['retry_after']} 秒后重试"
            )
        
        # 将限流信息添加到请求头
        request.state.rate_limit_info = info
        
        return info

# 使用示例
rate_limit_middleware = RateLimitMiddleware(requests_per_minute=60, requests_per_hour=1000)

@app.middleware("http")
async def rate_limit_middleware_func(request: Request, call_next):
    try:
        rate_limit_info = await rate_limit_middleware(request)
        response = await call_next(request)
        
        # 添加限流头信息
        response.headers["X-RateLimit-Remaining-Minute"] = str(rate_limit_info["minute_remaining"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(rate_limit_info["hour_remaining"])
        
        return response
        
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers={"Retry-After": str(rate_limit_info.get("retry_after", 60))}
        )
```

这个开发指南提供了完整的开发流程和最佳实践，帮助开发者快速理解和扩展MolEnc项目。