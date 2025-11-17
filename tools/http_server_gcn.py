"""Minimal HTTP server for GCN embeddings (mocked).

Run: python tools/http_server_gcn.py --port 8003
"""

import json
from typing import List
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

def encode_gcn(smiles_list: List[str], dim: int = 256):
    # Placeholder: deterministic hash-based vector for illustration
    out = []
    for s in smiles_list:
        h = abs(hash(s))
        rng = np.random.default_rng(h % (2**32))
        vec = rng.random(dim).astype(float).tolist()
        out.append(vec)
    return out


class Handler(BaseHTTPRequestHandler):
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
            dim = int(body.get('hidden_dim', 256))
            try:
                enc = encode_gcn(smiles, dim=dim)
                return self._json(200, {'encodings': enc})
            except Exception as e:
                return self._json(500, {'error': str(e)})

        if self.path == '/encode/single':
            smiles = body.get('smiles')
            dim = int(body.get('hidden_dim', 256))
            if not isinstance(smiles, str):
                return self._json(400, {'error': 'smiles_required'})
            try:
                enc = encode_gcn([smiles], dim=dim)
                return self._json(200, {'encoding': enc[0]})
            except Exception as e:
                return self._json(500, {'error': str(e)})

        return self._json(404, {'error': 'not_found'})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8003)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"GCN HTTP server listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == '__main__':
    main()