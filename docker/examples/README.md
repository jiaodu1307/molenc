# MolEnc Docker 示例代码

本目录包含MolEnc Docker化分子编码器的示例代码和测试工具。

## 文件说明

### 测试脚本

#### `test_local.py` - 本地API测试
模拟API接口逻辑的本地测试工具。

```bash
python docker/examples/test_local.py
```

**功能**:
- 验证API响应格式
- 测试Morgan编码器接口
- 测试ChemBERTa编码器接口
- 生成详细的测试报告

#### `integration_test.py` - 集成测试
验证端到端功能和错误处理的集成测试。

```bash
python docker/examples/integration_test.py
```

**功能**:
- API响应格式验证
- SMILES验证逻辑测试
- 批处理功能测试
- 错误处理机制验证
- 性能指标测试
- 数据完整性检查

#### `api_test.py` - API测试工具
完整的API测试工具，支持并发和负载测试。

```bash
# 基础测试
python docker/examples/api_test.py

# 并发测试
python docker/examples/api_test.py --concurrent 10

# 负载测试
python docker/examples/api_test.py --load-test --duration 60

# 指定服务
python docker/examples/api_test.py --service chemberta --smiles "CCO" "c1ccccc1"
```

### 客户端示例

#### `client_example.py` - Python客户端
演示如何使用MolEnc API的Python客户端示例。

```bash
# 基础使用演示
python docker/examples/client_example.py --demo basic

# 批处理演示
python docker/examples/client_example.py --demo batch

# DataFrame集成演示
python docker/examples/client_example.py --demo dataframe

# 错误处理演示
python docker/examples/client_example.py --demo error

# 性能对比演示
python docker/examples/client_example.py --demo performance
```

## 快速开始

### 1. 本地测试（无需Docker）

```bash
# 运行本地API测试
cd /home/jiaodu/projects/molenc
python docker/examples/test_local.py

# 运行集成测试
python docker/examples/integration_test.py
```

### 2. Docker环境测试

```bash
# 启动Docker服务
cd /home/jiaodu/projects/molenc/docker/scripts
./quickstart.sh

# 等待服务启动完成
# 运行API测试
cd /home/jiaodu/projects/molenc
docker exec molenc-chemberta python /app/examples/api_test.py
```

### 3. 使用客户端

```python
from client_example import MolEncClient

# 创建客户端
client = MolEncClient(base_url="http://localhost")

# 编码SMILES
smiles = ["CCO", "c1ccccc1", "CC(=O)O"]

# Morgan指纹
fingerprints = client.encode_morgan(smiles)
print("Morgan指纹:", fingerprints)

# ChemBERTa嵌入
embeddings = client.encode_chemberta(smiles)
print("ChemBERTa嵌入:", embeddings)
```

## 测试报告

测试完成后会生成详细的JSON格式报告：

- `test_report_local.json` - 本地API测试报告
- `integration_test_report.json` - 集成测试报告

报告包含：
- 测试摘要（通过率、响应时间）
- 详细测试结果
- 性能指标
- 错误信息

## 性能基准

### 响应时间目标
- 健康检查: < 1ms
- 信息接口: < 10ms
- 编码接口: < 100ms (单分子)
- 批处理: < 1s (32分子)

### 吞吐量目标
- 单分子编码: > 1000/秒
- 批处理: > 32000/秒 (32分子批次)

## 故障排除

### 常见问题

1. **Docker未安装**
   - 安装Docker和Docker Compose
   - 运行本地测试验证逻辑

2. **端口冲突**
   - 检查端口80、8001、8002是否被占用
   - 修改docker-compose.yml中的端口映射

3. **内存不足**
   - ChemBERTa需要较多内存
   - 减少批处理大小或启用GPU支持

4. **模型下载失败**
   - 检查网络连接
   - 手动下载模型到cache目录

### 调试建议

1. **查看日志**
   ```bash
   docker logs molenc-gateway
   docker logs molenc-morgan
   docker logs molenc-chemberta
   ```

2. **健康检查**
   ```bash
   curl http://localhost/health
   curl http://localhost/api/morgan/health
   curl http://localhost/api/chemberta/health
   ```

3. **性能监控**
   ```bash
   # 使用API测试工具
   python docker/examples/api_test.py --load-test --duration 60
   ```

## 扩展开发

### 添加新测试

1. 在测试脚本中添加新函数
2. 遵循现有的测试模式
3. 更新测试报告生成逻辑

### 自定义编码器

1. 参考现有编码器实现
2. 添加新的Dockerfile
3. 更新docker-compose.yml
4. 添加对应的测试用例

## 相关文档

- [部署指南](../docs/deployment_guide.md)
- [API参考](../docs/api_reference.md)
- [开发指南](../docs/development_guide.md)
- [测试总结](test_summary.md)

## 支持

如有问题，请参考：
- 部署指南中的故障排除部分
- API参考中的错误代码说明
- 开发指南中的架构说明