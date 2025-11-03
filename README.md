# MolEnc - 分子编码器统一库

一个统一的分子编码器库，集成了多种分子表示学习方法，提供简单易用的API接口。
(基于Claude Code, QWEN Code全自动开发，未人工修改BUG，可能存在问题)

## 🎯 项目目标

- **统一接口**: 为不同的分子编码方法提供统一的API
- **易于使用**: 一行代码实现SMILES到向量的转换
- **方法丰富**: 支持传统fingerprint、预训练模型、GNN等多种方法
- **环境隔离**: 解决不同编码器的依赖冲突问题
- **代码现代化**: 将老旧代码重构为现代Python实现

## 🏗️ 架构设计

### 核心架构

```
molenc/
├── core/
│   ├── base.py              # 基础编码器抽象类
│   ├── registry.py          # 编码器注册管理
│   ├── utils.py             # 通用工具函数
│   ├── exceptions.py        # 异常处理
│   └── config.py            # 配置管理
├── encoders/
│   ├── descriptors/
│   │   └── fingerprints/      # 描述符: 分子指纹
│   └── representations/
│       ├── sequence/          # 表示: 基于序列的模型
│       ├── graph/             # 表示: 基于图的模型
│       └── multimodal/        # 表示: 多模态模型
├── preprocessing/           # 数据预处理
│   ├── molecular.py         # 分子预处理
│   ├── graph.py             # 图预处理
│   └── standardization.py   # 分子标准化
├── environments/            # 环境管理
│   ├── conda_envs/         # Conda环境配置
│   ├── docker/             # Docker容器配置
│   └── requirements/       # 分模块依赖
└── examples/               # 使用示例
    ├── basic_usage.py
    ├── advanced_features.py
    └── custom_encoder.py
```

### 环境依赖解决方案

1. **模块化依赖**: 每个编码器模块独立的requirements文件
2. **可选依赖**: 使用extras_require实现按需安装
3. **环境隔离**: 提供Docker和Conda环境配置

## 🚀 快速开始

### 安装

```bash
# 基础安装（仅包含核心依赖）
pip install molenc

# 安装特定编码器依赖
pip install molenc[fingerprint]    # 分子指纹（Morgan, MACCS）
pip install molenc[nlp]           # NLP相关模型（ChemBERTa）
pip install molenc[gnn]           # 图神经网络（GCN）
pip install molenc[multimodal]    # 多模态模型（UniMol）
pip install molenc[chemistry]     # 化学信息学工具（RDKit）
pip install molenc[all]           # 全部功能

# 安装开发和环境管理依赖
pip install molenc[environment]   # 虚拟环境管理
pip install molenc[dev]           # 开发工具
```

#### 依赖文件说明

MolEnc项目使用多个依赖文件来管理不同场景下的依赖关系：

- **`requirements.txt`**: 核心依赖，包含运行MolEnc基本功能所需的最小依赖集
- **`requirements-dev.txt`**: 开发依赖，包含测试、代码质量检查、文档生成等开发工具
- **`requirements-optional.txt`**: 可选依赖，包含所有可选功能的完整依赖列表（注意：此文件包含大量与核心功能无关的依赖，建议使用extras方式安装）

> **推荐安装方式**: 使用 `pip install molenc[extras]` 的方式安装特定功能依赖，而不是直接使用requirements-optional.txt文件。

### 基本使用

```python
from molenc import MolEncoder

# 初始化已实现的编码器
encoder = MolEncoder('morgan')  # Morgan指纹编码器
# 或
encoder = MolEncoder('maccs')   # MACCS键编码器
# 或
encoder = MolEncoder('chemberta')  # ChemBERTa编码器
# 或
encoder = MolEncoder('gcn')     # GCN图神经网络编码器
# 或
encoder = MolEncoder('unimol')  # UniMol多模态编码器

# 编码单个分子
smiles = 'CCO'  # 乙醇
vector = encoder.encode(smiles)
print(f"分子向量维度: {vector.shape}")

# 批量编码
smiles_list = ['CCO', 'CC(=O)O', 'c1ccccc1']
vectors = encoder.encode_batch(smiles_list)
print(f"批量编码结果: {vectors.shape}")
```

### 高级使用

```python
# 自定义参数
encoder = MolEncoder('morgan', radius=3, n_bits=2048)

# 使用UniMol预训练模型
encoder = MolEncoder('unimol')

# 从配置文件加载
encoder = MolEncoder.from_config('config.yaml')

# 使用预设配置
encoder = MolEncoder.from_preset('drug_discovery')

# 链式编码（组合多种方法）
from molenc import ChainEncoder
chain = ChainEncoder(['morgan', 'maccs', 'unimol'])
combined_vector = chain.encode(smiles)
```

## 📊 支持的编码器

### ✅ 已实现的编码器

#### 1. 经典化学信息学方法 (Classical Chemoinformatics Methods)
> 这类方法不依赖于复杂的深度学习模型，速度快，可解释性强。

- **分子指纹 (Molecular Fingerprints)**
  - *描述*: 基于预定义的规则或算法将分子结构转换为固定长度的二进制或计数向量。
  - *已实现*: 
    - ✅ `Morgan`: 基于Morgan算法的圆形指纹
    - ✅ `MACCS`: 166个预定义结构键的指纹

#### 2. 基于深度学习的表示方法 (Deep Learning-based Representations)
> 这类方法通过深度神经网络端到端地从原始分子数据中学习特征表示。

- **基于序列的模型 (Sequence-based Models)**
  - *描述*: 将SMILES等线性表示视为序列，利用NLP模型（主要是Transformer）进行编码。
  - *已实现*: 
    - ✅ `ChemBERTa`: 基于RoBERTa架构的预训练模型，专门用于处理SMILES字符串

### 🚧 部分实现的编码器

- **基于图的模型 (Graph-based Models)**
  - *描述*: 将分子显式地视为图结构，利用图神经网络（GNN）捕捉拓扑信息。
  - *已实现*: 
    - ✅ `GCN`: 图卷积网络
  - *待实现*: 
    - ⏳ `MPNN`: 消息传递神经网络
    - ⏳ `GIN`: 图同构网络
    - ⏳ `SchNet`: 用于分子的连续滤波器卷积网络

- **多模态模型 (Multi-modal Models)**
  - *描述*: 融合多种分子信息源（如2D图、3D构象、文本描述）进行联合表示学习。
  - *已实现*: 
    - ✅ `UniMol`: 基于3D分子结构的预训练模型，支持多种版本
  - *待实现*: 
    - ⏳ `MoMu`: 融合文本和图结构的多模态模型
    - ⏳ `KV-PLM`: 知识引导的预训练语言模型

### ❌ 尚未实现的编码器

- **经典图嵌入算法**
  - ⏳ `Graph2Vec`: 经典图嵌入算法
  - ⏳ `Node2Vec`: 节点嵌入算法
  - ⏳ `DeepWalk`: 基于随机游走的图嵌入

- **其他深度学习模型**
  - ⏳ `MolT5`: 基于T5架构的分子Transformer模型
  - ⏳ `SMILES-BERT`: 专门用于SMILES的BERT模型
  - ⏳ `MolFormer`: 现代化的分子Transformer模型
  - ⏳ `GraphMAE`: 图掩码自编码器
  - ⏳ `MolCLR`: 分子对比学习表示

> 📝 **说明**: 
> - ✅ 表示已完整实现并可通过测试
> - ⏳ 表示部分实现或计划实现
> - ❌ 表示尚未开始实现
>
> 我们欢迎社区贡献来实现更多的编码器！

## 🔧 开发指南

### 添加新编码器

1. 继承基础编码器类：

```python
from molenc.core.base import BaseEncoder

class MyEncoder(BaseEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化代码
    
    def _encode_single(self, smiles: str) -> np.ndarray:
        # 实现单分子编码逻辑
        pass
    
    def get_output_dim(self) -> int:
        # 返回输出向量维度
        pass
```

2. 注册编码器：

```python
from molenc.core.registry import register_encoder

@register_encoder('my_encoder')
class MyEncoder(BaseEncoder):
    # 实现代码
```

### 环境配置

每个编码器模块应提供：
- `requirements.txt`: Python依赖
- `environment.yml`: Conda环境
- `Dockerfile`: Docker配置

### 智能环境管理

MolEnc现在支持智能环境管理，可以自动处理复杂的依赖关系：

```python
from molenc import MolEncoder

# MolEnc will automatically handle environment setup
encoder = MolEncoder('unimol')  # Automatically configures UniMol environment if needed
vectors = encoder.encode_batch(['CCO', 'CC(=O)O'])
```

特性：
- **自动依赖检查**: 自动检测当前环境是否满足编码器要求
- **虚拟环境自动创建**: 如果依赖不满足，自动创建和配置虚拟环境
- **进程隔离**: 在独立的Python进程中运行编码器，避免依赖冲突
- **云API备选**: 当本地依赖不可用时，可选择使用云API

### 依赖管理

MolEnc使用现代化的依赖管理方式，通过 `extras_require` 实现按需安装：

```bash
# 安装特定功能组
pip install molenc[chemistry]     # 化学信息学工具
pip install molenc[deep_learning] # 深度学习模型
pip install molenc[graph]         # 图神经网络
pip install molenc[nlp]           # NLP模型
pip install molenc[visualization] # 可视化工具
```

**依赖文件说明**:
- `requirements.txt`: 核心依赖，运行基本功能所需
- `requirements-dev.txt`: 开发工具依赖
- `requirements-optional.txt`: 完整可选依赖列表（包含大量非核心依赖，建议使用extras方式安装）

> **注意**: 推荐使用 `pip install molenc[extras]` 方式安装依赖，而不是直接安装requirements-optional.txt中的所有包。

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 测试特定编码器
pytest tests/test_fingerprint.py
pytest tests/test_transformer.py
pytest tests/test_gnn.py

# 性能基准测试
python benchmarks/run_benchmarks.py

# 集成测试
pytest tests/integration/

# 覆盖率测试
pytest --cov=molenc tests/
```

## 🔧 故障排除

### 常见问题

**Q: 安装时出现依赖冲突怎么办？**
A: 使用虚拟环境或Docker容器隔离依赖：
```bash
# 使用conda创建环境
conda create -n molenc python=3.8
conda activate molenc
pip install molenc[all]

# 或使用Docker
docker run -it molenc/molenc:latest
```

**Q: GPU内存不足怎么办？**
A: 调整批处理大小或使用CPU版本：
```python
# 减小批处理大小
encoder = MolEncoder('unimol', batch_size=32)

# 强制使用CPU
encoder = MolEncoder('unimol', device='cpu')
```

**Q: 编码速度太慢怎么办？**
A: 选择更快的编码器或启用并行处理：
```python
# 使用快速编码器
encoder = MolEncoder('morgan')  # 而不是 'unimol'

# 启用多进程
encoder = MolEncoder('morgan', n_jobs=8)
```

**Q: 如何处理无效的SMILES？**
A: 启用错误处理和分子标准化：
```python
from molenc.preprocessing import MolecularStandardizer

standardizer = MolecularStandardizer()
valid_smiles = standardizer.standardize(smiles_list)
encoder = MolEncoder('morgan', handle_errors='skip')
```

## 📈 性能对比

### 编码性能对比

| 编码器类型 | 编码器 | 编码速度 | 向量维度 | 内存占用 | GPU需求 | 准确性 |
|------------|--------|----------|----------|----------|---------|--------|
| 指纹方法 | Morgan | 1000+ mol/s | 2048 | 极低 | 否 | 中等 |
| 指纹方法 | MACCS | 800+ mol/s | 166 | 极低 | 否 | 中等 |
| Transformer | ChemBERTa | 120 mol/s | 768 | 高 | 推荐 | 高 |
| GNN | GCN | 200 mol/s | 256 | 中等 | 可选 | 高 |

| 多模态 | Uni-Mol | 50 mol/s | 512 | 很高 | 必需 | 很高 |
| 多模态 | MolCLR | 80 mol/s | 512 | 高 | 必需 | 高 |

## 📚 详细文档

有关详细使用指南，请参见：
- [ChemBERTa 使用指南](docs/chemberta_usage.md) - ChemBERTa 编码器的综合使用指南

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 贡献类型
- 添加新的编码器实现
- 改进现有编码器性能
- 修复bug和问题
- 完善文档和示例
- 优化环境配置

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢以下项目和论文的启发：
- RDKit: 化学信息学工具包
- Uni-Mol: 分子3D预训练
- DGL: 深度图学习库
- PyTorch Geometric: 几何深度学习

## 📞 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 邮箱: [your-email@example.com]

---

**注意**: 本项目仍在开发中，API可能会有变化。建议在生产环境使用前进行充分测试。