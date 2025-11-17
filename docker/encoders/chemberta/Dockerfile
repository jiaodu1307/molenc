FROM molenc-base:latest

# 安装GPU支持 (如果构建GPU版本)
ARG GPU_VERSION=false
RUN if [ "$GPU_VERSION" = "true" ]; then \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
    fi

# 安装ChemBERTa特定依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py ./

# 预下载模型 (可选，减少首次加载时间)
# RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1'); AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')"

HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

LABEL maintainer="MolEnc Team"
LABEL description="ChemBERTa Encoder"
LABEL version="1.0.0"

EXPOSE 8000