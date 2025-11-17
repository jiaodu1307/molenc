from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
import json
from molenc.encoders.descriptors.fingerprints.morgan import MorganEncoder

app = FastAPI(title="Morgan Fingerprint Encoder", version="1.0.0")

# 全局编码器实例
encoder = None

class EncodeRequest(BaseModel):
    smiles: Union[str, List[str]]
    radius: Optional[int] = 2
    n_bits: Optional[int] = 2048
    use_features: Optional[bool] = False

class EncodeResponse(BaseModel):
    fingerprints: List[List[float]]
    shape: List[int]
    metadata: dict

@app.on_event("startup")
async def startup_event():
    global encoder
    encoder = MorganEncoder()
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
            "use_features": "Use feature-based fingerprints (default: False)"
        }
    }

@app.post("/encode", response_model=EncodeResponse)
async def encode_molecules(request: EncodeRequest):
    try:
        # 确保smiles是列表
        smiles_list = [request.smiles] if isinstance(request.smiles, str) else request.smiles
        
        # 配置编码器参数 - MorganEncoder不支持动态参数更改，需要重新初始化
        global encoder
        encoder = MorganEncoder(
            radius=request.radius,
            n_bits=request.n_bits,
            use_features=request.use_features
        )
        
        # 编码分子
        fingerprints = encoder.encode(smiles_list)
        
        return EncodeResponse(
            fingerprints=fingerprints.tolist(),
            shape=list(fingerprints.shape),
            metadata={
                "n_molecules": len(smiles_list),
                "n_bits": request.n_bits,
                "radius": request.radius,
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
        
        # 配置编码器参数
        global encoder
        encoder = MorganEncoder(
            radius=request.radius,
            n_bits=request.n_bits,
            use_features=request.use_features
        )
        
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
                "radius": request.radius,
                "use_features": request.use_features,
                "batch_size": batch_size,
                "n_batches": len(range(0, len(smiles_list), batch_size))
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)