"""Reproducible closed-loop HTTP load, using only Python's standard library."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
from http.client import HTTPException
import json
import math
from pathlib import Path
import random
import re
import socket
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import uuid

MAX_BYTES = 10_485_760
MAX_RESPONSE_BYTES = 65_536
UUID = re.compile(r'[0-9a-f]{32}\Z')
ERROR_CODES = {'invalid_request','unsupported_media_type','image_too_large','invalid_image',
               'storage_unavailable','model_unavailable','inference_timeout','too_many_dice',
               'scene_too_complex','internal_error'}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write('\n')


def _read_input(path):
    with path.open('rb') as stream:
        body = stream.read(MAX_BYTES+1)
    if not body or len(body) > MAX_BYTES:
        raise ValueError('Input must be nonempty and within API byte limit')
    return body


def _inputs(manifest):
    path = Path(manifest).resolve()
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(document,dict):
        raise ValueError('Expected a manifest object')
    images = document.get('images')
    if not isinstance(images,list) or not images or len(images)>10000:
        raise ValueError('Expected 1..10000 input paths')
    result, seen = [], set()
    for name in images:
        if not isinstance(name,str):
            raise ValueError('Input path must be a string')
        source = (path.parent/name).resolve()
        if source in seen:
            raise ValueError('Duplicate input path')
        seen.add(source)
        mime = {'.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg'}.get(source.suffix.lower())
        if mime is None:
            raise ValueError('Input must have PNG/JPEG extension')
        body = _read_input(source)
        result.append({'path':source,'sha256':hashlib.sha256(body).hexdigest(),
                       'bytes':len(body),'content_type':mime})
    return sorted(result,key=lambda item:(item['sha256'],item['content_type']))


def _attempt(target, item, phase, sequence, timeout):
    event = {'schema_version':1,'phase':phase,'sequence':sequence,
             'timestamp_utc':datetime.now(timezone.utc).isoformat(),
             'input_sha256':item['sha256'],'input_bytes':item['bytes'],
             'content_type':item['content_type'],'sent':False,'http_status':None,
             'latency_ms':None,'request_id':None,'image_id':None,
             'status':None,'error_code':None,'client_error':None}
    try:
        body = _read_input(item['path'])
        if hashlib.sha256(body).hexdigest() != item['sha256']:
            event['client_error']='input_changed'
            return event
    except (OSError,ValueError):
        event['client_error']='input_unavailable'
        return event
    opener = build_opener(ProxyHandler({}),NoRedirect())
    req = Request(target,data=body,headers={'Content-Type':item['content_type']},method='POST')
    started = time.perf_counter()
    event['sent']=True
    try:
        try:
            response = opener.open(req,timeout=timeout)
        except HTTPError as exc:
            response = exc  # HTTP errors are observations, never retried.
        with response:
            event['http_status']=response.code
            raw = response.read(MAX_RESPONSE_BYTES+1)
        event['response_bytes']=len(raw)
        if len(raw)>MAX_RESPONSE_BYTES:
            event['client_error']='response_too_large'
        else:
            try:
                parsed=json.loads(raw)
                if not isinstance(parsed,dict):
                    raise ValueError()
                for key in ('request_id','image_id'):
                    value=parsed.get(key)
                    if isinstance(value,str) and UUID.fullmatch(value):
                        event[key]=value
                if parsed.get('status') in ('ok','needs_review','no_candidates'):
                    event['status']=parsed['status']
                error=parsed.get('error')
                if isinstance(error,dict) and error.get('code') in ERROR_CODES:
                    event['error_code']=error['code']
            except (ValueError,TypeError,RecursionError):
                event['client_error']='invalid_response'
    except (TimeoutError,socket.timeout):
        event['client_error']='timeout'
    except URLError as exc:
        event['client_error']='timeout' if isinstance(exc.reason,TimeoutError) else 'transport_error'
    except (OSError,ValueError,HTTPException):
        event['client_error']='transport_error'
    finally:
        event['latency_ms']=(time.perf_counter()-started)*1000
    return event


def run_load(manifest, output, *, base_url='http://127.0.0.1:24000', requests=10,
             seed=42, concurrency=1, warmup=1, timeout=45, progress=None):
    target=urlsplit(base_url)
    if (target.scheme not in ('http','https') or not target.hostname or target.username is not None
            or target.password is not None or target.query or target.fragment or target.path not in ('','/')):
        raise ValueError('Expected an HTTP origin without credentials or query')
    if (type(requests) is not int or requests<1 or type(warmup) is not int or warmup<0
            or type(concurrency) is not int or not 1<=concurrency<=16
            or not math.isfinite(timeout) or timeout<=0 or type(seed) is not int):
        raise ValueError('Invalid load settings')
    target_url=base_url.rstrip('/')+'/v1/recognize'
    images=_inputs(manifest)
    directory=Path(output)
    directory.mkdir(parents=True,exist_ok=False)
    public_inputs=[{key:value for key,value in item.items() if key!='path'} for item in images]
    run={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
         'target':target_url,'seed':seed,'requests':requests,'warmup':warmup,
         'concurrency':concurrency,'timeout_seconds':timeout,'inputs':public_inputs,
         'python':sys.version.split()[0],'retries':0,'redirects':False}
    _write_json(directory/'run.json',run)
    rng=random.Random(seed)
    measured, all_errors=[],0
    started=time.perf_counter()
    with (directory/'attempts.jsonl').open('x',encoding='utf-8') as stream:
        def record(event):
            nonlocal all_errors
            stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            if event['client_error'] or (event['http_status'] or 0)>=500:
                all_errors+=1
            if event['phase']=='measured':
                measured.append(event)
            if progress is not None:
                progress(event)

        # Separate RNG streams keep measured choices stable when warmup changes.
        warm_rng=random.Random(seed)
        for sequence in range(warmup):
            record(_attempt(target_url,warm_rng.choice(images),'warmup',sequence,timeout))
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending, sequence={},0
            while sequence<requests or pending:
                while sequence<requests and len(pending)<concurrency:
                    future=executor.submit(_attempt,target_url,rng.choice(images),'measured',sequence,timeout)
                    pending[future]=sequence
                    sequence+=1
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in sorted(done,key=lambda item:pending[item]):
                    record(future.result())
                    del pending[future]
    statuses=Counter(str(e['http_status']) for e in measured if e['http_status'] is not None)
    errors=Counter(e['client_error'] for e in measured if e['client_error'])
    latencies=sorted(e['latency_ms'] for e in measured if e['latency_ms'] is not None)
    percentile=lambda p:latencies[max(0,math.ceil(p*len(latencies))-1)] if latencies else None
    summary={'schema_version':1,'complete':True,'measured_attempts':len(measured),
             'warmup_attempts':warmup,'http_status_counts':dict(statuses),'client_error_counts':dict(errors),
             'sent_attempts':sum(e['sent'] for e in measured),'http_responses':sum(statuses.values()),
             'latency_samples':len(latencies),'latency_ms':{'p50':percentile(.5),'p95':percentile(.95),
                                                          'p99':percentile(.99)},
             'run_wall_seconds':time.perf_counter()-started,'failed_attempts_including_warmup':all_errors}
    _write_json(directory/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--base-url',default='http://127.0.0.1:24000')
    parser.add_argument('--requests',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--concurrency',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--timeout',type=float,default=45)
    args=parser.parse_args()
    output=args.output or Path('runtime/load')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    def progress(event):
        print(json.dumps(event,allow_nan=False),flush=True)
    try:
        summary=run_load(args.manifest,output,base_url=args.base_url,requests=args.requests,
                         seed=args.seed,concurrency=args.concurrency,warmup=args.warmup,
                         timeout=args.timeout,progress=progress)
    except KeyboardInterrupt:
        print('load_interrupted',file=sys.stderr)
        return 130
    except (OSError,ValueError,TypeError,RecursionError):
        print('load_configuration_or_output_error',file=sys.stderr)
        return 2
    print(json.dumps({'output':str(output),'summary':summary},allow_nan=False),flush=True)
    return 1 if summary['failed_attempts_including_warmup'] else 0


if __name__=='__main__':
    raise SystemExit(main())
