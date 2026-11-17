"""Read immutable ZIP sources and preserve each image as a queue entry."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image, ImageOps

TYPES = ('d4', 'd6', 'd8', 'd10', 'd10_tens', 'd12', 'd20')
TOKENS = ['0', '00', '1', '10', '11', '12', '13', '14', '15', '16',
          '17', '18', '19', '2', '20', '3', '30', '4', '40', '5',
          '50', '6', '60', '7', '70', '8', '80', '9', '90']
MAPS = {
    '200_dataset': ['d10', 'd10_tens', 'd12', 'd20', 'd6', 'd8'],
    'd4-turbo-rad-v4': ['d4', 'd4'],
    'dd-dice-detection': ['d10', 'd12', 'd6', 'd8'],
    'dice-number-detection': TOKENS,
}
EXPECTED_HASH = 'eeeb6c2873fc513bba6211a7df6d17ed33939cc0030b28a0da54d483f2a91562'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize_token(token):
    return None if token is None else (10 if token in ('0', '00') else int(token))


def instance(image_id, index, box, die_type=None, token=None, trusted=False,
             kind='body', polygon=None):
    return {
        'instance_id': f'{image_id}:{index}', 'bbox_xyxy': box, 'box_kind': kind,
        'polygon': polygon, 'die_type': die_type, 'face_token': token if trusted else None,
        'proposed_token': token if not trusted else None,
        'face_value': normalize_token(token) if trusted else None, 'type_known': die_type is not None,
        'value_known': trusted and token is not None, 'result_ambiguous': False,
        'occluded': False, 'truncated': False, 'modality': 'unknown',
        'convention': '', 'detection_known': kind == 'body',
    }


def orient_point(x, y, width, height, orientation):
    return {2: (width-x, y), 3: (width-x, height-y), 4: (x, height-y),
            5: (y, x), 6: (height-y, x), 7: (height-y, width-x),
            8: (y, width-x)}.get(orientation, (x, y))


def canonical_geometry(points, width, height, orientation):
    return [list(orient_point(x * width, y * height, width, height, orientation))
            for x, y in points]


def enclosing(points):
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def capture_labels(record, image_id, width, height, orientation):
    result = []
    for i, label in enumerate(record.get('labels', record.get('dice', []))):
        b = label.get('box')
        if not b:
            continue
        points = canonical_geometry([(b['x'], b['y']), (b['x']+b['w'], b['y']+b['h'])],
                                    width, height, orientation)
        kind = label.get('type', '').lower()
        kind = 'd10_tens' if kind == 'd100' else kind
        kind = kind if kind in TYPES else None
        value = label.get('value')
        token = str(value) if value is not None and str(value) in TOKENS else None
        trusted = record.get('valuesTrusted', record.get('values_trusted', False)) is True
        item = instance(image_id, i, enclosing(points), kind, token, trusted)
        item['source_annotation'] = label
        result.append(item)
    return result


def parse_yolo_source(archive, paths, name, image_id, width, height, orientation, source):
    parts = PurePosixPath(name).parts
    label_path = str(PurePosixPath(*parts[:-2], 'labels', PurePosixPath(name).stem + '.txt'))
    if label_path not in paths:
        return [], ['missing_annotations'], None
    text = archive.read(label_path).decode('utf-8-sig')
    labels, issues = [], []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            fields = line.split()
            cls = int(fields[0])
            vocab = MAPS[source]
            if cls < 0 or cls >= len(vocab):
                raise ValueError('unknown class')
            coords = list(map(float, fields[1:]))
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
                raise ValueError('invalid coordinates')
            polygon = None
            if len(coords) == 4:
                cx, cy, bw, bh = coords
                pts = [(cx-bw/2, cy-bh/2), (cx+bw/2, cy+bh/2)]
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                pts = list(zip(coords[::2], coords[1::2]))
                polygon = canonical_geometry(pts, width, height, orientation)
            else:
                raise ValueError('unsupported annotation geometry')
            pts = canonical_geometry(pts, width, height, orientation)
            numeral = source == 'dice-number-detection'
            labels.append(instance(image_id, i, enclosing(pts), None if numeral else vocab[cls],
                                   vocab[cls] if numeral else None, numeral,
                                   'numeral' if numeral else 'body', polygon))
            labels[-1]['source_class'] = cls
            labels[-1]['source_format'] = 'polygon' if polygon else 'yolo_box'
        except (ValueError, KeyError, IndexError) as exc:
            issues.append(f'annotation_line_{i+1}:{exc}')
    if not labels:
        issues.append('empty_annotations_unconfirmed')
    return labels, issues, label_path


def import_200_dataset(*args):
    return parse_yolo_source(*args, source='200_dataset')


def import_d4_dataset(*args):
    return parse_yolo_source(*args, source='d4-turbo-rad-v4')


def import_mixed_geometry_dataset(*args):
    return parse_yolo_source(*args, source='dd-dice-detection')


def import_numeral_dataset(*args):
    return parse_yolo_source(*args, source='dice-number-detection')


SOURCE_IMPORTERS = {
    '200_dataset': import_200_dataset,
    'd4-turbo-rad-v4': import_d4_dataset,
    'dd-dice-detection': import_mixed_geometry_dataset,
    'dice-number-detection': import_numeral_dataset,
}


def import_capture_manifest(archive):
    manifest = {}
    for line in archive.read('data/metadata.jsonl').decode('utf-8-sig').splitlines():
        record = json.loads(line)
        manifest['data/' + record['file_name'].removeprefix('data/')] = record
    return manifest


def import_final_sidecar(archive, paths, image_path, manifest_record):
    sidecar_path = f'raw/sidecars/{PurePosixPath(image_path).stem}.json'
    if sidecar_path in paths:
        sidecar = json.loads(archive.read(sidecar_path))
        if 'labels' in sidecar:
            if 'valuesTrusted' not in sidecar and manifest_record:
                sidecar['valuesTrusted'] = manifest_record.get('values_trusted', False)
            return sidecar, sidecar_path + '#labels'
    return manifest_record, 'data/metadata.jsonl' if manifest_record else None


def import_archive(archive_path: Path, cache_dir: Path, insert, expected_hash=EXPECTED_HASH):
    digest = sha256_file(archive_path)
    if digest != expected_hash.lower():
        raise ValueError(f'Archive SHA256 mismatch: {digest}')
    cache_dir.mkdir(parents=True, exist_ok=True)
    inventory = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        paths = {info.filename for info in infos}
        if len(paths) != len(infos):
            raise ValueError('Duplicate ZIP paths cannot have stable identities')
        manifest = import_capture_manifest(archive)
        for info in infos:
            name = info.filename
            raw = archive.read(info)
            content_hash = hashlib.sha256(raw).hexdigest()
            entry = {'path': name, 'sha256': content_hash, 'bytes': len(raw)}
            if PurePosixPath(name).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                inventory.append(entry)
                continue
            image_id = hashlib.sha256(name.encode()).hexdigest()[:24]
            source = '/'.join(PurePosixPath(name).parts[:2]) if name.startswith('roboflow/') else name.split('/')[0]
            issues, labels = [], []
            width = height = 0
            orientation = 1
            cache = None
            try:
                with Image.open(io.BytesIO(raw)) as original:
                    original.load()
                    ow, oh = original.size
                    orientation = original.getexif().get(274, 1)
                    canonical = ImageOps.exif_transpose(original).convert('RGB')
                    width, height = canonical.size
                    cache = cache_dir / (image_id + '.png')
                    output = io.BytesIO()
                    canonical.save(output, format='PNG')
                    encoded = output.getvalue()
                    expected_cache = hashlib.sha256(encoded).hexdigest()
                    if not cache.exists() or sha256_file(cache) != expected_cache:
                        temp = cache.with_suffix('.tmp')
                        temp.write_bytes(encoded)
                        temp.replace(cache)
                    entry['cache_sha256'] = expected_cache
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                issues.append(f'unreadable_image:{exc}')
                ow = oh = 0
            annotation_source = None
            if cache and name.startswith(('data/', 'raw/frames/')):
                record, annotation_source = import_final_sidecar(archive, paths, name, manifest.get(name))
                if record is not None:
                    labels = capture_labels(record, image_id, ow, oh, orientation)
                    annotation_source = annotation_source or 'data/metadata.jsonl'
                else:
                    issues.append('missing_final_labels')
            elif cache and name.startswith('roboflow/'):
                labels, extra, annotation_source = SOURCE_IMPORTERS[PurePosixPath(name).parts[1]](
                    archive, paths, name, image_id, ow, oh, orientation)
                issues.extend(extra)
            if len(labels) > 8:
                issues.append('above_eight')
            if any(x['box_kind'] == 'numeral' for x in labels):
                issues.append('numeral_not_body')
            for label in labels:
                x1, y1, x2, y2 = label['bbox_xyxy']
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    issues.append('invalid_box')
            frame = {'image_id': image_id, 'path': name, 'source': source,
                     'sha256': content_hash, 'width': width, 'height': height,
                     'orientation': orientation, 'orientation_applied': True,
                     'cache_path': str(cache) if cache else None, 'instances': labels,
                     'annotation_source': annotation_source, 'issues': sorted(set(issues)),
                     'decision': 'pending', 'reason': '', 'note': '', 'revision': 0,
                     'archive_sha256': digest, 'guide_version': 'audit-v1'}
            insert(frame)
            entry.update({'image_id': image_id, 'width': width, 'height': height,
                          'orientation': orientation, 'issues': frame['issues']})
            inventory.append(entry)
    return {'archive_path': str(archive_path), 'archive_sha256': digest,
            'files': len(inventory), 'images': sum('image_id' in row for row in inventory),
            'entries': inventory}
