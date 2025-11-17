from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import numpy as np
from maccs_encoder import MACCSEncoder

app = FastAPI(title="MACCS Keys Encoder", version="1.0.0")

# 全局编码器实例
encoder = None

class EncodeRequest(BaseModel):
    smiles: Union[str, List[str]]

class EncodeResponse(BaseModel):
    fingerprints: List[List[float]]
    shape: List[int]
    metadata: dict

@app.on_event("startup")
async def startup_event():
    global encoder
    encoder = MACCSEncoder()
    print("MACCS encoder initialized")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "encoder": "maccs", "version": "1.0.0"}

@app.get("/info")
async def get_info():
    return {
        "encoder": "maccs",
        "description": "MACCS keys molecular fingerprint encoder",
        "parameters": {
            "n_bits": "167-bit binary fingerprint",
            "features": "Structural key-based fingerprint capturing important molecular substructures"
        }
    }

@app.post("/encode", response_model=EncodeResponse)
async def encode_molecules(request: EncodeRequest):
    try:
        # 确保smiles是列表
        smiles_list = [request.smiles] if isinstance(request.smiles, str) else request.smiles
        
        # 编码分子
        fingerprints = encoder.encode(smiles_list)
        
        return EncodeResponse(
            fingerprints=fingerprints.tolist(),
            shape=list(fingerprints.shape),
            metadata={
                "n_molecules": len(smiles_list),
                "n_bits": encoder.n_bits,
                "encoder": "maccs"
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
        batch_size = 5000  # MACCS计算较快，可以使用较大的批大小
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
                "n_bits": encoder.n_bits,
                "batch_size": batch_size,
                "n_batches": len(range(0, len(smiles_list), batch_size)),
                "encoder": "maccs"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)