from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_TG = True
except Exception:
    HAS_TG = False

from rdkit import Chem

app = FastAPI()


class BatchRequest(BaseModel):
    smiles_list: List[str]
    hidden_dim: Optional[int] = 256
    model: Optional[str] = "gcn"


class SingleRequest(BaseModel):
    smiles: str
    hidden_dim: Optional[int] = 256
    model: Optional[str] = "gcn"


def atom_features(atom) -> List[float]:
    return [
        float(atom.GetAtomicNum()) / 100.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetTotalNumHs()) / 4.0,
        float(atom.GetDegree()) / 4.0,
        float(atom.GetImplicitValence()) / 6.0,
    ]


def mol_to_graph(smiles: str) -> Optional[Data]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
        edges = []
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx(); j = b.GetEndAtomIdx()
            edges.append([i, j]); edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)
    except Exception:
        return None


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, batch_graphs: List[Data]) -> torch.Tensor:
        outs = []
        for g in batch_graphs:
            if g.edge_index.numel() == 0:
                h = self.relu(self.conv1(g.x, torch.empty((2, 0), dtype=torch.long)))
                h = self.conv2(h, torch.empty((2, 0), dtype=torch.long))
            else:
                h = self.relu(self.conv1(g.x, g.edge_index))
                h = self.conv2(h, g.edge_index)
            pooled = h.mean(dim=0)
            outs.append(pooled)
        return torch.stack(outs, dim=0)


@app.get("/health")
def health():
    return {"status": "ok", "torch_geometric": HAS_TG}


@app.post("/encode/batch")
def encode_batch(req: BatchRequest):
    if not HAS_TG:
        return {"error": "torch-geometric not available"}
    hidden_dim = int(req.hidden_dim or 256)
    model = GCN(hidden_dim=hidden_dim)
    graphs = []
    for s in req.smiles_list:
        g = mol_to_graph(s)
        if g is None:
            g = Data(x=torch.zeros((1, 5)), edge_index=torch.empty((2, 0), dtype=torch.long))
        graphs.append(g)
    with torch.no_grad():
        vecs = model(graphs).cpu().numpy().astype(float).tolist()
    return {"encodings": vecs}


@app.post("/encode/single")
def encode_single(req: SingleRequest):
    batch = BatchRequest(smiles_list=[req.smiles], hidden_dim=req.hidden_dim, model=req.model)
    result = encode_batch(batch)
    if "encodings" in result:
        return {"encoding": result["encodings"][0]}
    return result