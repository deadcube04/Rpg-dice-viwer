"""Transactional, append-only audit revisions with optimistic concurrency."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import closing, contextmanager
import json
import math
from pathlib import Path
import re
import sqlite3

from dice_audit.importers import TYPES, TOKENS, normalize_token

# Temporary audit scope. Originals, revisions and full snapshots remain intact.
IGNORED_SOURCES = ('roboflow/d4-turbo-rad-v4',)
ACTIVE_SCOPE = 'source NOT IN (' + ','.join('?' for _ in IGNORED_SOURCES) + ')'
DARK_COLOR_WORD = r'(?:preto|preta|pretos|pretas|escuro|escura|escuros|escuras|negro|negra|negros|negras|black|dark)'
DARK_COLOR = re.compile(r'\b' + DARK_COLOR_WORD + r'\b', re.IGNORECASE)
DIE_TYPE = re.compile(r'\b(?:d4|d6|d8|d10|d12|d20|d10_tens)\b', re.IGNORECASE)
DARK_DIE_NOUN = re.compile(
    r'\bdados?\b.{0,25}\b' + DARK_COLOR_WORD + r'\b|\b' + DARK_COLOR_WORD + r'\b.{0,25}\bdados?\b',
    re.IGNORECASE)


def has_dark_die_observation(note):
    if not isinstance(note, str):
        return 0
    for sentence in re.split(r'[.;!?\n]+', note):
        if DARK_COLOR.search(sentence) and (DIE_TYPE.search(sentence) or DARK_DIE_NOUN.search(sentence)):
            return 1
    return 0


class Conflict(Exception):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS frames (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, hash TEXT NOT NULL,
                    document TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
                    decision TEXT NOT NULL DEFAULT 'pending');
                CREATE INDEX IF NOT EXISTS frames_hash ON frames(hash);
                CREATE TABLE IF NOT EXISTS revisions (
                    image_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL, document TEXT NOT NULL,
                    PRIMARY KEY(image_id, revision));
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.create_function('has_dark_die_observation', 1, has_dark_die_observation, deterministic=True)
        try:
            with db:
                yield db
        finally:
            db.close()

    def insert(self, frame):
        with self.connect() as db:
            row = db.execute('SELECT hash FROM frames WHERE id=?', (frame['image_id'],)).fetchone()
            if row and row['hash'] != frame['sha256']:
                raise ValueError('Image identity changed; use a separate database')
            db.execute('INSERT OR IGNORE INTO frames(id,source,hash,document) VALUES (?,?,?,?)',
                       (frame['image_id'], frame['source'], frame['sha256'], json.dumps(frame)))
            # Re-import may refresh imported proposals, never a manual revision.
            db.execute('UPDATE frames SET document=? WHERE id=? AND revision=0',
                       (json.dumps(frame), frame['image_id']))

    def get(self, image_id):
        with self.connect() as db:
            row = db.execute('SELECT document FROM frames WHERE id=?', (image_id,)).fetchone()
            if not row:
                raise KeyError(image_id)
            return json.loads(row['document'])

    def queue(self, query):
        conditions, params = [ACTIVE_SCOPE], list(IGNORED_SOURCES)
        for key, column in [('source', 'source'), ('decision', 'decision')]:
            if query.get(key):
                conditions.append(column + '=?')
                params.append(query[key])
        if query.get('double_check') == 'true' and not query.get('decision'):
            conditions.append("decision IN ('pending','reviewed')")
        if query.get('search'):
            conditions.append('(id LIKE ? OR json_extract(document,\'$.path\') LIKE ?)')
            params.extend(['%' + query['search'] + '%'] * 2)
        if query.get('count'):
            count = query['count']
            if count == '9+':
                conditions.append("json_array_length(document,'$.instances')>8")
            else:
                conditions.append("json_array_length(document,'$.instances')=?")
                params.append(int(count))
        if query.get('type'):
            conditions.append("EXISTS(SELECT 1 FROM json_each(document,'$.instances') WHERE json_extract(value,'$.die_type')=?)")
            params.append(query['type'])
        if query.get('issue'):
            conditions.append("EXISTS(SELECT 1 FROM json_each(document,'$.issues') WHERE value=?)")
            params.append(query['issue'])
        if query.get('double_check') == 'true':
            conditions.append("""(
                EXISTS(SELECT 1 FROM json_each(document,'$.instances')
                       WHERE json_extract(value,'$.die_type') IN ('d10','d10_tens'))
                OR EXISTS(SELECT 1 FROM json_each(document,'$.instances')
                          WHERE json_extract(value,'$.die_type') IN ('d10','d12','d20')
                            AND json_extract(value,'$.face_value') IN (6,9))
                OR EXISTS(SELECT 1 FROM json_each(document,'$.instances')
                          WHERE json_extract(value,'$.die_type') IN ('d8','d10','d12','d20')
                            AND json_extract(value,'$.face_value') IN (1,7))
                OR has_dark_die_observation(COALESCE(json_extract(document,'$.note'),''))=1
            )""")
        where = ' WHERE ' + ' AND '.join(conditions) if conditions else ''
        offset = max(0, int(query.get('offset', 0)))
        limit = min(100, max(1, int(query.get('limit', 30))))
        with self.connect() as db:
            total = db.execute('SELECT count(*) FROM frames' + where, params).fetchone()[0]
            rows = db.execute('SELECT document FROM frames' + where + ' ORDER BY source,id LIMIT ? OFFSET ?',
                              [*params, limit, offset]).fetchall()
        return {'items': [json.loads(row['document']) for row in rows], 'total': total, 'offset': offset}

    def progress(self):
        with self.connect() as db:
            rows = db.execute('SELECT source,decision,count(*) AS count FROM frames GROUP BY source,decision').fetchall()
            last = db.execute("SELECT id FROM frames WHERE id=(SELECT value FROM settings WHERE key='last_image') AND " + ACTIVE_SCOPE, IGNORED_SOURCES).fetchone()
            issues = db.execute("SELECT DISTINCT value FROM frames,json_each(document,'$.issues') WHERE " + ACTIVE_SCOPE + " ORDER BY value", IGNORED_SOURCES).fetchall()
        sources = {}
        total = reviewed = 0
        for row in rows:
            if row['source'] in IGNORED_SOURCES:
                continue
            source = sources.setdefault(row['source'], {'total': 0, 'reviewed': 0})
            source['total'] += row['count']
            total += row['count']
            if row['decision'] != 'pending':
                source['reviewed'] += row['count']
                reviewed += row['count']
        return {'total': total, 'reviewed': reviewed, 'sources': sources,
                'inventory_total': sum(row['count'] for row in rows),
                'ignored_sources': list(IGNORED_SOURCES),
                'last_image': last[0] if last else None, 'issues': [row[0] for row in issues],
                'approved': False}

    def resume(self, image_id):
        self.get(image_id)
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', ('last_image', image_id))

    def duplicates(self, image_id):
        frame = self.get(image_id)
        with self.connect() as db:
            rows = db.execute('SELECT document FROM frames WHERE hash=? AND id<>? AND ' + ACTIVE_SCOPE + ' ORDER BY id',
                              (frame['sha256'], image_id, *IGNORED_SOURCES)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def history(self, image_id):
        self.get(image_id)
        with self.connect() as db:
            rows = db.execute('SELECT revision,created_at,document FROM revisions WHERE image_id=? ORDER BY revision DESC',
                              (image_id,)).fetchall()
        return [{'revision': row[0], 'created_at': row[1], 'document': json.loads(row[2])} for row in rows]

    def save(self, image_id, payload):
        if not isinstance(payload, dict):
            raise ValueError('Objeto de revisão obrigatório')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT document,revision FROM frames WHERE id=?', (image_id,)).fetchone()
            if not row:
                raise KeyError(image_id)
            if type(payload.get('revision')) is not int or payload['revision'] != row['revision']:
                raise Conflict('A revisão mudou. Recarregue a imagem antes de salvar; seu rascunho foi preservado.')
            frame = json.loads(row['document'])
            validate_review(payload, frame)
            for key in ('decision', 'reason', 'note', 'instances'):
                frame[key] = payload[key]
            frame['revision'] += 1
            frame['reviewed_at'] = datetime.now(timezone.utc).isoformat()
            # New UI does not ask for a person's identity; old history stays intact.
            frame.pop('reviewer', None)
            document = json.dumps(frame, ensure_ascii=False)
            # Preserve the initial imported document before the first manual edit.
            if row['revision'] == 0:
                db.execute('INSERT INTO revisions VALUES (?,?,?,?)',
                           (image_id, 0, frame['reviewed_at'], row['document']))
            db.execute('INSERT INTO revisions VALUES (?,?,?,?)',
                       (image_id, frame['revision'], frame['reviewed_at'], document))
            db.execute('UPDATE frames SET document=?,revision=?,decision=? WHERE id=?',
                       (document, frame['revision'], frame['decision'], image_id))
        return frame

    def backup(self, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source, closing(sqlite3.connect(destination)) as target:
            source.backup(target)

    def snapshot(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT document FROM frames ORDER BY id')]


def validate_review(payload, frame):
    if not isinstance(payload, dict):
        raise ValueError('Objeto de revisão obrigatório')
    if payload.get('decision') not in ('pending', 'reviewed', 'negative', 'excluded'):
        raise ValueError('Decisão inválida')
    for key in ('reason', 'note'):
        if not isinstance(payload.get(key), str) or len(payload[key]) > 10000:
            raise ValueError(f'Campo textual inválido: {key}')
    if payload['decision'] == 'excluded' and not payload['reason'].strip():
        raise ValueError('Exclusão exige motivo')
    labels = payload.get('instances')
    if not isinstance(labels, list) or len(labels) > 1000:
        raise ValueError('Lista de instâncias inválida')
    if payload['decision'] == 'negative' and labels:
        raise ValueError('Ausência confirmada exige zero instâncias')
    if payload['decision'] == 'reviewed' and not labels:
        raise ValueError('Confirme ausência de dados ou exclua com motivo')
    ids = set()
    for label in labels:
        if not isinstance(label, dict):
            raise ValueError('Instância deve ser um objeto')
        identity = label.get('instance_id')
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('ID de instância inválido ou repetido')
        ids.add(identity)
        box = label.get('bbox_xyxy')
        if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
            raise ValueError('Caixa inválida')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= frame['width'] and 0 <= y1 < y2 <= frame['height']):
            raise ValueError('Caixa fora da imagem')
        if label.get('die_type') not in (*TYPES, None) or label.get('face_token') not in (*TOKENS, None):
            raise ValueError('Tipo ou token inválido')
        for flag in ('type_known', 'value_known', 'result_ambiguous', 'occluded', 'truncated', 'detection_known'):
            if type(label.get(flag)) is not bool:
                raise ValueError(f'Máscara inválida: {flag}')
        if label['type_known'] != (label['die_type'] is not None):
            raise ValueError('Tipo desconhecido precisa ser nulo')
        if label['value_known'] and (label['face_token'] is None or label['result_ambiguous']):
            raise ValueError('Valor conhecido exige token e resultado não ambíguo')
        if label.get('box_kind') not in ('body', 'numeral'):
            raise ValueError('Diferencie caixa de corpo e numeral')
        if label['detection_known'] and label['box_kind'] != 'body':
            raise ValueError('Caixa de numeral não supervisiona o detector')
        if label.get('modality') not in ('unknown', 'pips', 'numerals'):
            raise ValueError('Modalidade inválida')
        if not isinstance(label.get('convention'), str):
            raise ValueError('Convenção inválida')
        # Untrusted proposals are preserved separately, never canonical truth.
        if not label['value_known']:
            if label['face_token'] is not None:
                label['proposed_token'] = label['face_token']
            label['face_token'] = None
        label['face_value'] = normalize_token(label['face_token'])
        if label['value_known'] and label['die_type']:
            token, kind = label['face_token'], label['die_type']
            valid = (token in ['00', '10', '20', '30', '40', '50', '60', '70', '80', '90']
                     if kind == 'd10_tens' else token != '00' and
                     (token == '0' and kind == 'd10' or 1 <= int(token) <= int(kind[1:])))
            if not valid:
                raise ValueError('Token incompatível com o tipo; corrija ou marque ambiguidade')
