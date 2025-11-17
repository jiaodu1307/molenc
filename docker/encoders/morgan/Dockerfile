FROM molenc-base:latest

# Morgan指纹编码器 - 无额外依赖
# 基础镜像已包含RDKit和所有必需依赖

# 复制应用代码
COPY app.py ./

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 标签
LABEL maintainer="MolEnc Team"
LABEL description="Morgan Fingerprint Encoder"
LABEL version="1.0.0"

EXPOSE 8000