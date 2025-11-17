"""
SchNet Encoder FastAPI Service
Provides REST API for SchNet molecular encoder
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

from schnet_encoder import SchNetEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global encoder instance
encoder: Optional[SchNetEncoder] = None

class MoleculeRequest(BaseModel):
    """Request model for single molecule encoding"""
    smiles: str = Field(..., description="SMILES string of the molecule")
    conformations: Optional[int] = Field(default=1, description="Number of conformations to generate")

class BatchMoleculeRequest(BaseModel):
    """Request model for batch molecule encoding"""
    molecules: List[str] = Field(..., description="List of SMILES strings")
    conformations: Optional[int] = Field(default=1, description="Number of conformations per molecule")

class EncodingResponse(BaseModel):
    """Response model for encoding results"""
    embeddings: List[List[float]] = Field(..., description="Molecular embeddings")
    dimensions: int = Field(..., description="Dimensionality of embeddings")
    processing_time: float = Field(..., description="Processing time in seconds")
    molecule_count: int = Field(..., description="Number of molecules processed")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    encoder_loaded: bool = Field(..., description="Whether encoder is loaded")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")

class InfoResponse(BaseModel):
    """Response model for service info"""
    name: str = Field(..., description="Encoder name")
    version: str = Field(..., description="Encoder version")
    description: str = Field(..., description="Encoder description")
    dimensions: int = Field(..., description="Output dimensions")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    supported_features: List[str] = Field(..., description="Supported features")

# Service metadata
SERVICE_VERSION = "1.0.0"
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global encoder
    
    # Startup
    logger.info("Starting SchNet encoder service...")
    try:
        encoder = SchNetEncoder()
        logger.info("SchNet encoder loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load SchNet encoder: {e}")
        encoder = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down SchNet encoder service...")
    encoder = None

# Create FastAPI app
app = FastAPI(
    title="SchNet Molecular Encoder API",
    description="REST API for SchNet 3D molecular encoder",
    version=SERVICE_VERSION,
    lifespan=lifespan
)

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "SchNet Molecular Encoder API",
        "version": SERVICE_VERSION,
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    return HealthResponse(
        status="healthy" if encoder is not None else "degraded",
        encoder_loaded=encoder is not None,
        version=SERVICE_VERSION,
        uptime=uptime
    )

@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Get encoder information"""
    if encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Encoder not loaded"
        )
    
    return InfoResponse(
        name="SchNet",
        version="1.0.0",
        description="SchNet 3D molecular encoder with continuous-filter convolutional neural networks",
        dimensions=encoder.get_output_dim(),
        max_sequence_length=512,
        supported_features=["3d_coordinates", "multiple_conformations", "batch_processing"]
    )

@app.post("/encode", response_model=EncodingResponse)
async def encode_molecule(request: MoleculeRequest):
    """Encode a single molecule"""
    if encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Encoder not loaded"
        )
    
    try:
        start_time_proc = time.time()
        
        # Generate embedding for single molecule
        embedding_vec = encoder.encode_smiles(
            request.smiles,
            max_conformers=request.conformations
        )
        
        processing_time = time.time() - start_time_proc
        
        # Wrap single embedding as list of list for schema consistency
        embeddings_list = [embedding_vec.tolist()]
        
        return EncodingResponse(
            embeddings=embeddings_list,
            dimensions=len(embeddings_list[0]) if embeddings_list else 0,
            processing_time=processing_time,
            molecule_count=1
        )
        
    except Exception as e:
        logger.error(f"Error encoding molecule: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to encode molecule: {str(e)}"
        )

@app.post("/encode_batch", response_model=EncodingResponse)
async def encode_batch(request: BatchMoleculeRequest):
    """Encode multiple molecules in batch"""
    if encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Encoder not loaded"
        )
    
    if not request.molecules:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No molecules provided"
        )
    
    try:
        start_time_proc = time.time()
        
        # Generate embeddings for all molecules
        all_embeddings: List[List[float]] = []
        for smiles in request.molecules:
            try:
                rep = encoder.encode_smiles(
                    smiles,
                    max_conformers=request.conformations
                )
                all_embeddings.append(rep.tolist())
            except Exception as e:
                logger.warning(f"Failed to encode {smiles}: {e}")
                # Use zero vector for failed molecules to keep batch shape
                all_embeddings.append([0.0] * encoder.get_output_dim())
        
        processing_time = time.time() - start_time_proc
        
        return EncodingResponse(
            embeddings=all_embeddings,
            dimensions=len(all_embeddings[0]) if all_embeddings else 0,
            processing_time=processing_time,
            molecule_count=len(request.molecules)
        )
        
    except Exception as e:
        logger.error(f"Error encoding batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to encode batch: {str(e)}"
        )

@app.post("/encode_stream")
async def encode_stream(request: BatchMoleculeRequest):
    """Stream encoding results for large batches"""
    if encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Encoder not loaded"
        )
    
    if not request.molecules:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No molecules provided"
        )
    
    try:
        results = []
        for i, smiles in enumerate(request.molecules):
            try:
                rep = encoder.encode_smiles(
                    smiles,
                    max_conformers=request.conformations
                )
                results.append({
                    "index": i,
                    "smiles": smiles,
                    "embeddings": rep.tolist(),
                    "success": True
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "smiles": smiles,
                    "error": str(e),
                    "success": False
                })
        
        return JSONResponse(content={
            "results": results,
            "total": len(request.molecules),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"])
        })
    except Exception as e:
        logger.error(f"Error during stream encoding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Stream encoding failed: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    logger.info(f"Starting SchNet API server on {host}:{port}")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )