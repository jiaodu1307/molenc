"""Minimal HTTP server for ChemBERTa embeddings.

Run: python tools/http_server_chemberta.py --port 8002
"""

import json
from typing import List
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TF = True
except Exception:
    HAS_TF = False


class ChemBERTaService:
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        if not HAS_TF:
            raise RuntimeError("transformers/torch not available in server")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.hidden = self.model.config.hidden_size

    def encode(self, smiles_list: List[str], pooling: str = "cls") -> List[List[float]]:
        inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # [B, L, H]
        if pooling == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            # mean pooling over non-padded tokens
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.cpu().numpy().astype(float).tolist()


def make_service(model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
    return ChemBERTaService(model_name)


class Handler(BaseHTTPRequestHandler):
    service = None

    def _json(self, code: int, payload: dict):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        data = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
        try:
            body = json.loads(data)
        except Exception:
            return self._json(400, {'error': 'invalid_json'})

        if self.path == '/encode/batch':
            smiles = body.get('smiles_list') or body.get('smiles') or []
            if isinstance(smiles, str):
                smiles = [smiles]
            model_name = body.get('model_name') or 'seyonec/ChemBERTa-zinc-base-v1'
            pooling = body.get('pooling_strategy') or 'cls'
            try:
                if Handler.service is None:
                    Handler.service = make_service(model_name)
                enc = Handler.service.encode(smiles, pooling)
                return self._json(200, {'encodings': enc})
            except Exception as e:
                return self._json(500, {'error': str(e)})

        if self.path == '/encode/single':
            smiles = body.get('smiles')
            model_name = body.get('model_name') or 'seyonec/ChemBERTa-zinc-base-v1'
            pooling = body.get('pooling_strategy') or 'cls'
            if not isinstance(smiles, str):
                return self._json(400, {'error': 'smiles_required'})
            try:
                if Handler.service is None:
                    Handler.service = make_service(model_name)
                enc = Handler.service.encode([smiles], pooling)
                return self._json(200, {'encoding': enc[0]})
            except Exception as e:
                return self._json(500, {'error': str(e)})

        return self._json(404, {'error': 'not_found'})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8002)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"ChemBERTa HTTP server listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == '__main__':
    main()