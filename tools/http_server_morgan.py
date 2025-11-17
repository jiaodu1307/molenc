"""Minimal HTTP server for Morgan fingerprint encoding.

Run: python tools/http_server_morgan.py --port 8000
"""

import json
from typing import List
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

import numpy as np


def encode_morgan(smiles_list: List[str], radius: int = 2, n_bits: int = 2048):
    if not HAS_RDKIT:
        raise RuntimeError("RDKit not available in server")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out.append([0] * n_bits)
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        for bid in fp.GetOnBits():
            if bid < n_bits:
                arr[bid] = 1
        out.append(arr.astype(int).tolist())
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
            radius = int(body.get('radius', 2))
            n_bits = int(body.get('n_bits', 2048))
            try:
                enc = encode_morgan(smiles, radius=radius, n_bits=n_bits)
                return self._json(200, {'encodings': enc})
            except Exception as e:
                return self._json(500, {'error': str(e)})

        if self.path == '/encode/single':
            smiles = body.get('smiles')
            radius = int(body.get('radius', 2))
            n_bits = int(body.get('n_bits', 2048))
            if not isinstance(smiles, str):
                return self._json(400, {'error': 'smiles_required'})
            try:
                enc = encode_morgan([smiles], radius=radius, n_bits=n_bits)
                return self._json(200, {'encoding': enc[0]})
            except Exception as e:
                return self._json(500, {'error': str(e)})

        return self._json(404, {'error': 'not_found'})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"Morgan HTTP server listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == '__main__':
    main()