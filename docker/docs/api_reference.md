# MolEnc API 参考文档

本文档详细描述了MolEnc Docker服务的所有API端点、请求参数和响应格式。

## 基础信息

### API版本
- **版本**: v1.0
- **基础URL**: `http://localhost`
- **网关端口**: 80 (HTTP), 8080 (管理接口)
- **编码器端口**: 8001 (Morgan), 8002 (ChemBERTa)

### 认证
当前版本不需要认证，所有端点都是公开访问的。

### 数据格式
- **请求格式**: JSON
- **响应格式**: JSON
- **字符编码**: UTF-8

### 通用响应格式
```json
{
  "success": true,
  "message": "操作成功",
  "data": { ... },
  "metadata": { ... }
}
```

## 网关API

### 健康检查
检查整个系统的健康状态。

**端点**: `GET /health`

**请求参数**: 无

**响应示例**:
```json
{
  "success": true,
  "message": "系统运行正常",
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "services": {
      "gateway": "healthy",
      "morgan": "healthy",
      "chemberta": "healthy"
    }
  },
  "metadata": {
    "version": "1.0.0",
    "uptime": 3600
  }
}
```

**状态码**:
- 200: 系统健康
- 503: 系统不健康

### 根路径
获取系统信息。

**端点**: `GET /`

**响应示例**:
```json
{
  "success": true,
  "message": "欢迎使用MolEnc分子编码器",
  "data": {
    "name": "MolEnc Molecular Encoder",
    "version": "1.0.0",
    "description": "Docker化的分子编码器服务",
    "endpoints": {
      "morgan": "/api/morgan",
      "chemberta": "/api/chemberta",
      "health": "/health"
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Morgan指纹编码器API

### 健康检查
检查Morgan编码器服务状态。

**端点**: `GET /api/morgan/health`

**响应示例**:
```json
{
  "success": true,
  "message": "Morgan编码器运行正常",
  "data": {
    "status": "healthy",
    "encoder": "morgan",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "metadata": {
    "version": "1.0.0"
  }
}
```

### 获取编码器信息
获取Morgan编码器的配置信息和参数说明。

**端点**: `GET /api/morgan/info`

**响应示例**:
```json
{
  "success": true,
  "message": "Morgan编码器信息",
  "data": {
    "encoder": "morgan",
    "description": "Morgan指纹编码器，基于RDKit实现",
    "parameters": {
      "n_bits": {
        "type": "integer",
        "default": 1024,
        "range": [64, 16384],
        "description": "指纹位数"
      },
      "radius": {
        "type": "integer",
        "default": 2,
        "range": [1, 5],
        "description": "指纹半径"
      },
      "use_features": {
        "type": "boolean",
        "default": false,
        "description": "是否使用特征信息"
      }
    },
    "output_format": {
      "type": "list of lists",
      "description": "每个分子对应的指纹向量"
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 编码SMILES
将SMILES字符串编码为Morgan指纹。

**端点**: `POST /api/morgan/encode`

**请求体**:
```json
{
  "smiles": ["CCO", "c1ccccc1"],
  "n_bits": 1024,
  "radius": 2,
  "use_features": false
}
```

**参数说明**:
| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| smiles | array[string] | 是 | - | SMILES字符串数组 |
| n_bits | integer | 否 | 1024 | 指纹位数 (64-16384) |
| radius | integer | 否 | 2 | 指纹半径 (1-5) |
| use_features | boolean | 否 | false | 是否使用特征信息 |

**响应示例**:
```json
{
  "success": true,
  "message": "编码成功",
  "data": {
    "fingerprints": [
      [0, 1, 0, 1, 1, 0, ...],
      [1, 0, 1, 0, 0, 1, ...]
    ],
    "shape": [2, 1024]
  },
  "metadata": {
    "encoder": "morgan",
    "n_bits": 1024,
    "radius": 2,
    "use_features": false,
    "processing_time": 0.123,
    "n_molecules": 2
  }
}
```

**错误响应**:
```json
{
  "success": false,
  "message": "无效的SMILES字符串",
  "error": {
    "type": "ValueError",
    "details": "无法解析SMILES: 'invalid_smiles'"
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

**状态码**:
- 200: 编码成功
- 400: 请求参数错误
- 422: 数据验证错误
- 500: 服务器内部错误

### 批量编码
批量编码SMILES字符串（功能与编码接口相同，但优化了大批量处理）。

**端点**: `POST /api/morgan/encode_batch`

**请求体**: 同编码接口

**响应格式**: 同编码接口

## ChemBERTa编码器API

### 健康检查
检查ChemBERTa编码器服务状态。

**端点**: `GET /api/chemberta/health`

**响应示例**:
```json
{
  "success": true,
  "message": "ChemBERTa编码器运行正常",
  "data": {
    "status": "healthy",
    "encoder": "chemberta",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "metadata": {
    "version": "1.0.0"
  }
}
```

### 获取编码器信息
获取ChemBERTa编码器的配置信息和参数说明。

**端点**: `GET /api/chemberta/info`

**响应示例**:
```json
{
  "success": true,
  "message": "ChemBERTa编码器信息",
  "data": {
    "encoder": "chemberta",
    "description": "ChemBERTa分子嵌入编码器，基于Transformer模型",
    "parameters": {
      "model_name": {
        "type": "string",
        "default": "seyonec/ChemBERTa-zinc-base-v1",
        "options": ["seyonec/ChemBERTa-zinc-base-v1", "seyonec/ChemBERTa-zinc250k-v1"],
        "description": "预训练模型名称"
      },
      "max_length": {
        "type": "integer",
        "default": 512,
        "range": [128, 1024],
        "description": "最大序列长度"
      },
      "pooling_strategy": {
        "type": "string",
        "default": "mean",
        "options": ["cls", "mean", "max"],
        "description": "池化策略"
      }
    },
    "output_format": {
      "type": "list of lists",
      "description": "每个分子对应的嵌入向量"
    },
    "model_info": {
      "hidden_size": 768,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "vocab_size": 28996
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 编码SMILES
将SMILES字符串编码为ChemBERTa嵌入向量。

**端点**: `POST /api/chemberta/encode`

**请求体**:
```json
{
  "smiles": ["CCO", "c1ccccc1"],
  "model_name": "seyonec/ChemBERTa-zinc-base-v1",
  "max_length": 512,
  "pooling_strategy": "mean"
}
```

**参数说明**:
| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| smiles | array[string] | 是 | - | SMILES字符串数组 |
| model_name | string | 否 | "seyonec/ChemBERTa-zinc-base-v1" | 预训练模型名称 |
| max_length | integer | 否 | 512 | 最大序列长度 (128-1024) |
| pooling_strategy | string | 否 | "mean" | 池化策略 (cls/mean/max) |

**响应示例**:
```json
{
  "success": true,
  "message": "编码成功",
  "data": {
    "embeddings": [
      [0.123, -0.456, 0.789, ...],
      [-0.234, 0.567, -0.890, ...]
    ],
    "shape": [2, 768]
  },
  "metadata": {
    "encoder": "chemberta",
    "model_name": "seyonec/ChemBERTa-zinc-base-v1",
    "max_length": 512,
    "pooling_strategy": "mean",
    "processing_time": 0.456,
    "n_molecules": 2
  }
}
```

**错误响应**: 同Morgan编码器

**状态码**: 同Morgan编码器

### 批量编码
批量编码SMILES字符串（功能与编码接口相同，但优化了大批量处理）。

**端点**: `POST /api/chemberta/encode_batch`

**请求体**: 同编码接口

**响应格式**: 同编码接口

## 管理接口

管理接口运行在8080端口，提供系统管理功能。

### 系统状态
获取详细的系统状态信息。

**端点**: `GET /status`

**响应示例**:
```json
{
  "success": true,
  "message": "系统状态信息",
  "data": {
    "system": {
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "disk_usage": 23.4,
      "uptime": 86400
    },
    "services": {
      "morgan": {
        "status": "running",
        "cpu_usage": 12.3,
        "memory_usage": 256,
        "request_count": 1000,
        "error_rate": 0.01
      },
      "chemberta": {
        "status": "running",
        "cpu_usage": 34.5,
        "memory_usage": 2048,
        "request_count": 500,
        "error_rate": 0.02
      }
    }
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 服务重启
重启指定的服务。

**端点**: `POST /restart/{service}`

**路径参数**:
- `service`: 服务名称 (morgan/chemberta)

**响应示例**:
```json
{
  "success": true,
  "message": "服务重启成功",
  "data": {
    "service": "morgan",
    "status": "restarted",
    "restart_time": "2024-01-15T10:30:00Z"
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## 错误处理

### 错误响应格式
所有错误响应都遵循以下格式：

```json
{
  "success": false,
  "message": "错误描述",
  "error": {
    "type": "错误类型",
    "details": "详细错误信息",
    "trace_id": "错误追踪ID"
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### 常见错误码

| HTTP状态码 | 错误类型 | 描述 |
|------------|----------|------|
| 400 | BadRequest | 请求参数错误 |
| 404 | NotFound | 资源不存在 |
| 422 | ValidationError | 数据验证失败 |
| 429 | RateLimitExceeded | 请求频率超限 |
| 500 | InternalServerError | 服务器内部错误 |
| 503 | ServiceUnavailable | 服务不可用 |

### 错误示例

#### 无效的SMILES
```json
{
  "success": false,
  "message": "无效的SMILES字符串",
  "error": {
    "type": "ValueError",
    "details": "无法解析SMILES: 'invalid_smiles'",
    "trace_id": "abc123"
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### 参数验证错误
```json
{
  "success": false,
  "message": "参数验证失败",
  "error": {
    "type": "ValidationError",
    "details": "n_bits必须在64到16384之间",
    "field": "n_bits",
    "value": 32
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### 服务不可用
```json
{
  "success": false,
  "message": "服务暂时不可用",
  "error": {
    "type": "ServiceUnavailable",
    "details": "ChemBERTa编码器正在初始化模型",
    "retry_after": 30
  },
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## 性能考虑

### 请求限制
- **最大分子数**: 1000 (单次请求)
- **最大请求体大小**: 10MB
- **超时时间**: 300秒
- **并发限制**: 每个编码器最多10个并发请求

### 优化建议

#### 批量处理
```python
# 推荐：批量处理
smiles_list = ["CCO", "c1ccccc1", "CCCO", ...]  # 大批量
response = requests.post("/api/morgan/encode", json={"smiles": smiles_list})

# 不推荐：单个处理
for smiles in smiles_list:
    response = requests.post("/api/morgan/encode", json={"smiles": [smiles]})
```

#### 错误处理
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置重试策略
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("http://", adapter)
session.mount("https://", adapter)

# 使用会话发送请求
try:
    response = session.post("/api/morgan/encode", json={"smiles": smiles_list})
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
```

## 版本历史

- **v1.0.0**: 初始版本
  - 支持Morgan指纹编码
  - 支持ChemBERTa嵌入编码
  - 提供健康检查和状态监控
  - 支持批量处理

- **v1.0.1**: 性能优化
  - 优化批处理性能
  - 添加并发请求限制
  - 改进错误处理

- **v1.0.2**: 管理功能
  - 添加系统状态监控
  - 支持服务重启
  - 添加性能统计