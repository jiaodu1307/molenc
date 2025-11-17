## 现状问题审视
- 编码器注册/懒加载存在设计缺陷，影响可插拔性与快速扩展：
  - 预注册将 `encoder_class=None` 写入注册表，导致不会触发懒加载而直接失败（`molenc/core/registry.py:46-56`, `molenc/core/registry.py:224-249`）。
  - 模块级注册虽有 `@register_encoder('morgan')`（`molenc/encoders/descriptors/fingerprints/morgan.py:14-16`），但若未显式导入，将被前述缺陷掩盖。
- API/工厂与 CLI 不一致，用户路径分裂：
  - 同时存在复杂工厂与简化工厂/接口（`molenc/core/encoder_factory.py`, `molenc/core/simple_encoder_factory.py`, `molenc/simple_api.py:15-31`），依赖管理与隔离能力不一致，CLI 默认导入也不稳定（`molenc/cli.py:15-28`）。
- 依赖就绪检测粗糙，难以跨环境快速部署：
  - 仅通过 `__import__` 判断包可用，缺少版本、CUDA/GPU、构建通道（pip/conda/mamba）等约束（`molenc/core/smart_encoder_selector.py:164-172`）。
  - 高级依赖管理虽有虚拟环境与自动安装接口，但默认不启用且与选择器/工厂耦合松散（`molenc/environments/advanced_dependency_manager.py:113-145`, `molenc/core/encoder_factory.py:97-121`）。
- 环境隔离支持不完整：
  - 仅支持 venv/子进程，conda、Docker/K8s、远程服务为可选/半成品，缺少统一后端抽象（`molenc/isolation/smart_environment_manager.py:235-252`）。
- 缺少跨技术路线的插件契约：
  - 新增非 Python 编码器（如 Go/Java、微服务）没有标准协议/注册途径；返回结构与批量流式约定不足。
- 配置系统仅关注模型参数，缺少环境/后端维度：
  - 预设无法声明后端与依赖层级，难以一键切换部署策略（`molenc/core/config.py:1-41`）。

## 影响与根因
- 可插拔/懒加载失效 → 新增编码器需要手动导入或修改核心代码，阻碍快速集成。
- 路径分裂/默认禁用自动安装 → 用户需要理解多个入口与开关，降低“开箱即用”。
- 检测不可靠/环境后端缺失 → 在不同 OS/包管理器/硬件场景下容易失败或表现不稳定。
- 缺少统一后端抽象 → 无法将 Python 本地、conda、Docker、HTTP/gRPC 置于同一接口下进行策略选择。

## 改进目标
- 建立稳定、统一的“编码器提供者”抽象，支持本地/隔离/容器/远程多后端无缝切换。
- 修复注册/懒加载，使新增编码器零改核心代码、可声明式注册。
- 强化依赖就绪与版本约束，内建自动安装与回退链。
- 让 CLI/API 的默认路径“可用”：一条命令可在不同环境策略下成功运行。

## 实施方案（分阶段）
### 阶段 1：修复关键缺陷并统一入口
- 修改注册表：
  - 引入 `register_module(name, module_path)` 或允许 `encoder_class` 可为空时仅写入 `_encoder_modules`；`get_encoder` 在 `encoder_class is None` 时触发懒加载（修正 `molenc/core/registry.py:31-75`）。
  - 移除将 `None` 写入 `_encoders` 的逻辑（修正 `molenc/core/registry.py:224-249`）。
- 统一使用高级工厂于 CLI 与简单 API：
  - `molenc/cli.py` 和 `molenc/simple_api.py` 调用 `EncoderFactory`，并提供 `--mode`、`--auto-install`、`--backend` 等显式参数。

### 阶段 2：后端抽象与集成
- 新增 `ExecutionBackend` 抽象与实现：`LocalPython`、`VenvSubprocess`、`CondaEnv`、`DockerContainer`、`HttpGrpcRemote`。
- 在 `EncoderFactory` 中选择后端并透传到构造器，配合 `SmartEnvironmentManager` 管理生命周期。
- 统一数据契约：请求/响应 JSON schema，支持批量与流式，numpy/torch 张量序列化策略。

### 阶段 3：依赖就绪与安装策略
- 扩展检测：版本范围、可选特性（CUDA/AVX）、二进制兼容；采用 `packaging.specifiers` 与 `importlib.metadata`。
- 安装策略：pip/conda/mamba 命令建议与自动执行；后端感知（在 conda/Docker 内安装）。
- 选择器策略增强：引入“后端可用性 + 依赖等级”作为评分维度（扩展 `molenc/core/smart_encoder_selector.py`）。

### 阶段 4：容器与远程
- 提供官方轻量镜像（RDKit 指纹、ChemBERTa CPU 版），统一健康检查端点与 `/encode` 协议。
- CLI 支持 `--backend docker` 与 `--backend http`，自动拉取镜像/连接远程。

### 阶段 5：配置与预设重构
- 为 `Config` 增加 `backend`、`dependency_level`、`auto_install`、`fallback_chain` 字段（扩展 `molenc/core/config.py`）。
- 提供“场景预设”：`local_minimal`、`conda_compat`、`docker_stable`、`cloud_fallback`。

### 阶段 6：一致性与验证
- 建立跨后端一致性测试：同一 SMILES 在不同后端的指纹/嵌入误差阈值校验。
- 性能/可靠性基准集成（复用 `molenc/core/performance_comparator.py:323-381`）。

## 里程碑与交付
- M1（1 周）：注册修复 + CLI 统一参数 + 本地/venv 后端落地。
- M2（2 周）：Conda/Docker 后端与依赖策略；两款官方镜像发布。
- M3（2 周）：远程 HTTP/gRPC 后端与插件协议；预设重构。
- M4（1 周）：一致性与性能基准、文档与示例。

## 风险与回退
- RDKit/深度学习依赖复杂 → 通过容器与 conda 提供“保底环境”。
- 远程接口安全与稳定性 → 默认只开放本地网络与令牌校验；提供重试与超时。

## 验证标准
- 新增编码器无需改核心，仅声明注册即可被 CLI/API 懒加载调用。
- 在 `local/conda/docker/http` 四种后端下，`molenc encode --encoder morgan` 均可一次成功（自动安装或回退链）。
- 基准报告展示不同后端的性能/可靠性差异，并给出默认策略推荐。