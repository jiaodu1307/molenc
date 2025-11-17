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
    # 使用默认参数初始化
    encoder = ChemBERTaEncoder(
        model_name="seyonec/ChemBERTa-zinc-base-v1",
        max_length=512,
        pooling_strategy="mean"
    )
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
        
        # 配置编码器 - ChemBERTaEncoder不支持动态参数更改，需要重新初始化
        global encoder
        encoder = ChemBERTaEncoder(
            model_name=request.model_name,
            max_length=request.max_length,
            pooling_strategy=request.pooling_strategy
        )
        
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

@app.post("/encode/batch")
async def encode_batch(request: EncodeRequest):
    """批量编码接口"""
    try:
        smiles_list = [request.smiles] if isinstance(request.smiles, str) else request.smiles
        
        # 配置编码器
        global encoder
        encoder = ChemBERTaEncoder(
            model_name=request.model_name,
            max_length=request.max_length,
            pooling_strategy=request.pooling_strategy
        )
        
        # 分批处理大文件
        batch_size = 32  # ChemBERTa批大小较小以避免内存问题
        all_embeddings = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            emb = encoder.encode(batch)
            all_embeddings.append(emb)
        
        # 合并结果
        embeddings = np.vstack(all_embeddings)
        
        return EncodeResponse(
            embeddings=embeddings.tolist(),
            shape=list(embeddings.shape),
            metadata={
                "n_molecules": len(smiles_list),
                "embedding_dim": embeddings.shape[1],
                "batch_size": batch_size,
                "n_batches": len(range(0, len(smiles_list), batch_size)),
                "model_name": request.model_name,
                "pooling_strategy": request.pooling_strategy
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)