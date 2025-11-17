"""
D-MPNN Molecular Encoder API

This module provides a FastAPI service for the D-MPNN molecular encoder.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import time
import logging
from contextlib import asynccontextmanager

# Import the D-MPNN encoder
from dmpnn_encoder import DMPNNEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global encoder instance
encoder = None

# Request/Response models
class SingleEncodeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule")
    depth: Optional[int] = Field(3, description="Number of message passing steps")

class BatchEncodeRequest(BaseModel):
    molecules: List[str] = Field(..., description="List of SMILES strings")
    depth: Optional[int] = Field(3, description="Number of message passing steps")

class StreamEncodeRequest(BaseModel):
    molecules: List[str] = Field(..., description="List of SMILES strings")
    depth: Optional[int] = Field(3, description="Number of message passing steps")

class EncodeResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Molecular embeddings")
    dimensions: int = Field(..., description="Dimension of embeddings")
    processing_time: float = Field(..., description="Processing time in seconds")
    molecule_count: int = Field(..., description="Number of molecules processed")

class SingleEncodeResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Molecular embeddings")
    dimensions: int = Field(..., description="Dimension of embeddings")
    processing_time: float = Field(..., description="Processing time in seconds")
    molecule_count: int = Field(..., description="Number of molecules processed")

class StreamResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Individual encoding results")
    successful: int = Field(..., description="Number of successful encodings")
    failed: int = Field(..., description="Number of failed encodings")
    processing_time: float = Field(..., description="Total processing time in seconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    encoder_loaded: bool = Field(..., description="Whether encoder is loaded")
    timestamp: float = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")

class InfoResponse(BaseModel):
    name: str = Field(..., description="Encoder name")
    description: str = Field(..., description="Encoder description")
    dimensions: int = Field(..., description="Embedding dimensions")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    supported_features: List[str] = Field(..., description="Supported features")

# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global encoder
    
    # Startup
    logger.info("Starting D-MPNN encoder service...")
    try:
        encoder = DMPNNEncoder(
            hidden_size=300,
            depth=3,
            dropout=0.0,
            aggregation='mean'
        )
        logger.info("D-MPNN encoder loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load D-MPNN encoder: {e}")
        encoder = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down D-MPNN encoder service...")

# Create FastAPI app
app = FastAPI(
    title="D-MPNN Molecular Encoder API",
    description="Directed Message Passing Neural Network for molecular representation learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service start time
start_time = time.time()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "D-MPNN Molecular Encoder API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info", 
            "encode": "/encode",
            "encode_batch": "/encode_batch",
            "encode_stream": "/encode_stream"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        encoder_loaded=encoder is not None,
        timestamp=time.time(),
        version="1.0.0",
        uptime=time.time() - start_time
    )

@app.get("/info", response_model=InfoResponse)
async def get_info():
    """Get encoder information"""
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")
    
    return InfoResponse(
        name="D-MPNN Encoder",
        description="Directed Message Passing Neural Network for molecular representation learning",
        dimensions=encoder.get_embedding_dimension(),
        max_sequence_length=512,
        supported_features=["SMILES", "batch_processing", "streaming"]
    )

@app.post("/encode", response_model=SingleEncodeResponse)
async def encode_single(request: SingleEncodeRequest):
    """Encode a single molecule"""
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")
    
    start_time = time.time()
    
    try:
        # Encode the molecule
        embedding = encoder.encode_smiles(request.smiles)
        
        processing_time = time.time() - start_time
        
        return SingleEncodeResponse(
            embeddings=[embedding.tolist()],
            dimensions=len(embedding),
            processing_time=processing_time,
            molecule_count=1
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error encoding molecule: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/encode_batch", response_model=EncodeResponse)
async def encode_batch(request: BatchEncodeRequest):
    """Encode a batch of molecules"""
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")
    
    if not request.molecules:
        raise HTTPException(status_code=400, detail="Empty molecule list")
    
    if len(request.molecules) > 1000:
        raise HTTPException(status_code=400, detail="Too many molecules (max 1000)")
    
    start_time = time.time()
    
    try:
        # Encode batch
        embeddings = encoder.encode_batch(request.molecules)
        
        processing_time = time.time() - start_time
        
        return EncodeResponse(
            embeddings=embeddings.tolist(),
            dimensions=embeddings.shape[1],
            processing_time=processing_time,
            molecule_count=len(request.molecules)
        )
        
    except Exception as e:
        logger.error(f"Error encoding batch: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/encode_stream", response_model=StreamResponse)
async def encode_stream(request: StreamEncodeRequest):
    """Stream encoding of molecules"""
    if encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")
    
    if not request.molecules:
        raise HTTPException(status_code=400, detail="Empty molecule list")
    
    start_time = time.time()
    
    results = []
    successful = 0
    failed = 0
    
    for smiles in request.molecules:
        try:
            embedding = encoder.encode_smiles(smiles)
            results.append({
                "smiles": smiles,
                "success": True,
                "embeddings": embedding.tolist()
            })
            successful += 1
        except ValueError as e:
            results.append({
                "smiles": smiles,
                "success": False,
                "error": str(e)
            })
            failed += 1
        except Exception as e:
            logger.error(f"Error encoding {smiles}: {e}")
            results.append({
                "smiles": smiles,
                "success": False,
                "error": "Internal error"
            })
            failed += 1
    
    processing_time = time.time() - start_time
    
    return StreamResponse(
        results=results,
        successful=successful,
        failed=failed,
        processing_time=processing_time
    )

@app.get("/docs")
async def get_docs():
    """Get API documentation"""
    return {"message": "API documentation available at /docs (Swagger UI) or /redoc (ReDoc)"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)