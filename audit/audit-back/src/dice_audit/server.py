"""Loopback-only audit API and static UI. No models, training or MinIO."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import traceback
from urllib.parse import parse_qs, urlsplit
import zipfile

from PIL import Image

from dice_audit.store import Conflict, Store
from dice_audit.importers import import_archive

BACKEND_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = BACKEND_ROOT.parent


def main():
    parser = argparse.ArgumentParser(description='Auditoria manual completa; não executa treinamento')
    parser.add_argument('command', choices=['import', 'serve', 'backup', 'export', 'status'])
    parser.add_argument('--archive', type=Path, default=Path('C:/diecamera-frames/data.zip'))
    parser.add_argument('--runtime', type=Path, default=BACKEND_ROOT / 'runtime')
    parser.add_argument('--ui-dist', type=Path, default=AUDIT_ROOT / 'audit-ui/dist')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    args.runtime.mkdir(parents=True, exist_ok=True)
    store = Store(args.runtime / 'audit-v1.sqlite3')
    if args.command == 'status':
        summary = store.progress()
        sources = {}
        for frame in store.snapshot():
            row = sources.setdefault(frame['source'], {'images': 0, 'with_annotations': 0,
                'instances': 0, 'with_known_value': 0})
            row['images'] += 1
            row['with_annotations'] += frame['annotation_source'] is not None
            row['instances'] += len(frame['instances'])
            row['with_known_value'] += sum(label['value_known'] for label in frame['instances'])
        summary['structural_sources'] = sources
        print(json.dumps(summary, ensure_ascii=False))
        return
    if args.command == 'import':
        inventory = import_archive(args.archive, args.runtime / 'images', store.insert)
        path = args.runtime / 'inventory.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(inventory, ensure_ascii=False), encoding='utf8')
        temporary.replace(path)
        print(json.dumps({k: v for k, v in inventory.items() if k != 'entries'}))
        return
    if args.command == 'backup':
        print(backup(store, args.runtime))
        return
    if args.command == 'export':
        print(json.dumps(export(store, args.runtime)))
        return

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.route()

        def do_POST(self):
            self.route()

        def do_PUT(self):
            self.route()

        def route(self):
            try:
                self.handle_route()
            except Conflict as exc:
                self.send_json(409, {'detail': str(exc)})
            except KeyError:
                self.send_json(404, {'detail': 'Imagem não encontrada'})
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.send_json(400, {'detail': str(exc)})
            except Exception:
                traceback.print_exc()
                self.log_error('Erro interno na auditoria')
                self.send_json(500, {'detail': 'Falha interna; a revisão não foi confirmada'})

        def handle_route(self):
            target = urlsplit(self.path)
            path = target.path
            query = {k: values[0] for k, values in parse_qs(target.query).items()}
            if self.command != 'GET':
                # JSON writes cannot be initiated by cross-origin HTML forms.
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('Content-Type application/json obrigatório')
                origin = self.headers.get('Origin')
                if origin and origin not in (f'http://127.0.0.1:{args.port}', 'http://127.0.0.1:5173', 'http://localhost:5173'):
                    self.send_json(403, {'detail': 'Origem não permitida'})
                    return
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 2 * 1024 * 1024:
                    raise ValueError('Tamanho de revisão inválido')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('Objeto JSON obrigatório')
            if path == '/api/progress' and self.command == 'GET':
                self.send_json(200, store.progress())
            elif path == '/api/frames' and self.command == 'GET':
                self.send_json(200, store.queue(query))
            elif path == '/api/backup' and self.command == 'POST':
                self.send_json(201, {'path': str(backup(store, args.runtime))})
            elif path == '/api/exports' and self.command == 'POST':
                self.send_json(201, export(store, args.runtime))
            elif path.startswith('/api/frames/'):
                parts = path.strip('/').split('/')
                image_id = parts[2]
                action = parts[3] if len(parts) == 4 else ''
                frame = store.get(image_id)
                if self.command == 'PUT' and action == 'review':
                    self.send_json(200, store.save(image_id, payload))
                elif self.command == 'POST' and action == 'resume':
                    store.resume(image_id)
                    self.send_json(200, {'saved': True})
                elif self.command == 'GET' and action == 'history':
                    self.send_json(200, store.history(image_id))
                elif self.command == 'GET' and action == 'duplicates':
                    self.send_json(200, store.duplicates(image_id))
                elif self.command == 'GET' and action == 'image':
                    cache = resolve_cache(frame, args.runtime)
                    if cache is None:
                        self.send_json(404, {'detail': 'Imagem ilegível; exclua com motivo'})
                        return
                    if 'crop' in query:
                        index = int(query['crop'])
                        if not 0 <= index < len(frame['instances']):
                            raise ValueError('Instância inexistente')
                        with Image.open(cache) as image:
                            image = image.crop(frame['instances'][index]['bbox_xyxy'])
                            output = io.BytesIO()
                            image.save(output, format='PNG')
                            self.send_bytes(output.getvalue(), 'image/png')
                    else:
                        self.send_bytes(cache.read_bytes(), 'image/png')
                elif self.command == 'GET' and action == 'original':
                    inventory = json.loads((args.runtime / 'inventory.json').read_text(encoding='utf8'))
                    with zipfile.ZipFile(inventory['archive_path']) as archive:
                        body = archive.read(frame['path'])
                    if hashlib.sha256(body).hexdigest() != frame['sha256']:
                        raise ValueError('Original não corresponde ao hash inventariado')
                    self.send_bytes(body, mimetypes.guess_type(frame['path'])[0] or 'application/octet-stream')
                elif self.command == 'GET' and not action:
                    self.send_json(200, frame)
                else:
                    self.send_json(404, {'detail': 'Rota não encontrada'})
            elif path.startswith('/api/'):
                self.send_json(404, {'detail': 'Rota não encontrada'})
            elif self.command == 'GET':
                dist = args.ui_dist
                file = (dist / path.lstrip('/')).resolve()
                if not file.is_relative_to(dist.resolve()):
                    raise ValueError('Caminho inválido')
                if not file.is_file():
                    file = dist / 'index.html'
                if file.is_file():
                    self.send_bytes(file.read_bytes(), mimetypes.guess_type(str(file))[0] or 'application/octet-stream')
                else:
                    self.send_json(503, {'detail': 'Compile audit-ui com bun run build ou use o Vite na porta 5173'})
            else:
                self.send_json(404, {'detail': 'Rota não encontrada'})

        def send_bytes(self, body, content_type):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Auditoria local: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def backup(store, runtime):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = runtime / 'backups' / f'audit-{stamp}.sqlite3'
    store.backup(path)
    return path


def resolve_cache(frame, runtime):
    cache = frame.get('cache_path')
    if cache and Path(cache).is_file():
        return Path(cache)
    # Restored legacy backups retain absolute paths from before relocation.
    # Generated caches use stable image IDs, independent of the checkout path.
    image_id = frame.get('image_id', '')
    if not isinstance(image_id, str) or len(image_id) != 24 or any(c not in '0123456789abcdef' for c in image_id):
        return None
    canonical = runtime / 'images' / (image_id + '.png')
    return canonical if canonical.is_file() else None


def export(store, runtime):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory = runtime / 'exports' / stamp
    directory.mkdir(parents=True)
    frames = store.snapshot()
    body = ''.join(json.dumps(frame, ensure_ascii=False) + '\n' for frame in frames).encode()
    (directory / 'revisions.jsonl').write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    reviewed = sum(frame['decision'] != 'pending' for frame in frames)
    metadata = {'path': str(directory), 'frames': len(frames), 'reviewed': reviewed,
                'manifest_sha256': digest, 'approved': False,
                'purpose': 'audit_snapshot_not_training_dataset'}
    (directory / 'snapshot.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    backup(store, runtime)
    return metadata


if __name__ == '__main__':
    main()
