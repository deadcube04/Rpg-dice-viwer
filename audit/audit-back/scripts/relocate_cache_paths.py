"""Relocate operational cache paths without changing annotations or revisions."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3


def annotation_digest(connection):
    digest = hashlib.sha256()
    for table in ('frames', 'revisions'):
        for identity, raw in connection.execute(f'SELECT rowid,document FROM {table} ORDER BY rowid'):
            document = json.loads(raw)
            document.pop('cache_path', None)
            digest.update(json.dumps([table, identity, document], sort_keys=True).encode())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True, type=Path)
    parser.add_argument('--from-runtime', required=True, type=Path)
    parser.add_argument('--to-runtime', required=True, type=Path)
    args = parser.parse_args()
    if not args.database.is_file() or not args.to_runtime.is_dir():
        raise ValueError('Existing database and destination runtime required; stop the audit server first')
    old_root = args.from_runtime.resolve()
    new_root = args.to_runtime.resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup_path = new_root / 'backups' / f'before-relocation-{stamp}.sqlite3'
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(args.database)) as connection:
        with closing(sqlite3.connect(backup_path)) as target:
            connection.backup(target)
        before = annotation_digest(connection)
        changes = {}
        with connection:
            connection.execute('BEGIN IMMEDIATE')
            for table in ('frames', 'revisions'):
                changed = 0
                rows = connection.execute(f'SELECT rowid,document FROM {table}').fetchall()
                for identity, raw in rows:
                    document = json.loads(raw)
                    cache = document.get('cache_path')
                    if not cache:
                        continue
                    try:
                        relative = Path(cache).relative_to(old_root)
                    except ValueError:
                        continue
                    new_cache = new_root / relative
                    if not new_cache.is_file():
                        raise ValueError(f'Missing relocated cache: {new_cache}; transaction rolled back')
                    document['cache_path'] = str(new_cache)
                    connection.execute(f'UPDATE {table} SET document=? WHERE rowid=?',
                                       (json.dumps(document, ensure_ascii=False), identity))
                    changed += 1
                changes[table] = changed
            after = annotation_digest(connection)
            if before != after:
                raise ValueError('Annotation/history contents changed; transaction rolled back')
    report = {'database': str(args.database.resolve()), 'from_runtime': str(old_root),
              'to_runtime': str(new_root), 'backup': str(backup_path),
              'updated_cache_paths': changes, 'annotations_sha256': after,
              'annotations_preserved': True}
    report_dir = new_root / 'migrations'
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f'relocation-{stamp}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
