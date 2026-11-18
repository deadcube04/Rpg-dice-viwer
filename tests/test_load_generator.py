"""Real local HTTP boundary: no model, Docker or network service required."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dice_viewer.recognition.load import run_load


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'private-name.png').write_bytes(b'image-A')
        (self.root / 'other.jpg').write_bytes(b'image-B')
        self.manifest = self.root / 'inputs.json'
        self.manifest.write_text(json.dumps({'images': ['private-name.png','other.jpg']}))
        self.seen = []
        seen = self.seen
        owner = self
        self.on_request = None

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                seen.append(body)
                if owner.on_request is not None:
                    owner.on_request()
                if body == b'slow':
                    time.sleep(0.2)
                status = 429 if body == b'image-B' else 200
                if body == b'redirect':
                    status = 302
                payload = {'request_id': 'a'*32, 'image_id': 'b'*32, 'status': 'ok',
                           'private': 'PRIVATE_TOKEN'} if status == 200 else {'error': 'PRIVATE_TOKEN'}
                encoded = b'PRIVATE_TOKEN' if body == b'invalid' else json.dumps(payload).encode()
                if body == b'nested':
                    encoded = b'{"x":'*1500+b'0'+b'}'*1500
                if body == b'large':
                    encoded = (b'PRIVATE_TOKEN'*10)[:129]
                self.send_response(status)
                if status == 302:
                    self.send_header('Location','/unknown/PRIVATE_TOKEN')
                self.send_header('Content-Length',str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread = Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def run_case(self, name, **kwargs):
        output = self.root / name
        summary = run_load(self.manifest, output, base_url=self.base, requests=8,
                           warmup=1, seed=42, **kwargs)
        events = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()]
        return output, summary, events

    def test_same_seed_repeats_inputs_without_retries_and_separates_warmup(self):
        _, summary, first = self.run_case('one')
        _, _, second = self.run_case('two')
        ordered = lambda rows: [(r['phase'],r['sequence'],r['input_sha256']) for r in rows]
        self.assertEqual(ordered(first),ordered(second))
        self.assertEqual(len(self.seen),18)
        self.assertEqual(len(first),9)
        self.assertEqual(first[0]['phase'],'warmup')
        self.assertEqual(summary['measured_attempts'],8)
        self.assertEqual(sum(summary['http_status_counts'].values()),8)
        self.assertIn('429',summary['http_status_counts'])
        self.assertEqual(summary['client_error_counts'],{})
        self.assertTrue(all(r['latency_ms'] >= 0 for r in first))

    def test_outputs_do_not_copy_image_names_bodies_or_private_response_fields(self):
        output, _, events = self.run_case('safe')
        text = ''.join(p.read_text() for p in output.iterdir())
        for secret in ('PRIVATE_TOKEN','private-name.png','image-A','image-B'):
            self.assertNotIn(secret,text)
        success = next(e for e in events if e['http_status']==200)
        self.assertEqual(success['request_id'],'a'*32)
        self.assertEqual(success['image_id'],'b'*32)

    def test_refuses_existing_output_without_modifying_it(self):
        output = self.root/'existing'
        output.mkdir()
        (output/'keep').write_text('unchanged')
        with self.assertRaises(FileExistsError):
            run_load(self.manifest,output,base_url=self.base,requests=1)
        self.assertEqual((output/'keep').read_text(),'unchanged')
        self.assertEqual(self.seen,[])

    def test_concurrency_keeps_one_record_per_planned_attempt(self):
        _, summary, events = self.run_case('parallel',concurrency=3)
        measured = [e for e in events if e['phase']=='measured']
        self.assertEqual(sorted(e['sequence'] for e in measured),list(range(8)))
        self.assertEqual(len(events),9)
        self.assertEqual(summary['measured_attempts'],8)

    def test_rejects_urls_with_credentials_or_query_before_sending(self):
        for base in ('http://user:password@localhost:24000','http://localhost:24000/?token=PRIVATE'):
            with self.assertRaises(ValueError):
                run_load(self.manifest,self.root/'invalid',base_url=base,requests=1)
        self.assertEqual(self.seen,[])

    def test_timeout_has_one_attempt_and_no_http_response(self):
        (self.root/'slow.png').write_bytes(b'slow')
        self.manifest.write_text(json.dumps({'images':['slow.png']}))
        summary=run_load(self.manifest,self.root/'timeout',base_url=self.base,
                         requests=1,warmup=0,timeout=0.05)
        self.assertEqual(len(self.seen),1)
        self.assertEqual(summary['client_error_counts'],{'timeout':1})
        self.assertEqual(summary['http_responses'],0)
        self.assertEqual(summary['sent_attempts'],1)

    def test_malformed_response_is_observable_without_copying_response_text(self):
        (self.root/'invalid.png').write_bytes(b'invalid')
        self.manifest.write_text(json.dumps({'images':['invalid.png']}))
        output=self.root/'invalid-response'
        summary=run_load(self.manifest,output,base_url=self.base,requests=1,warmup=0)
        self.assertEqual(summary['client_error_counts'],{'invalid_response':1})
        self.assertNotIn('PRIVATE_TOKEN',(output/'attempts.jsonl').read_text())

    def test_changed_inputs_are_not_sent_again_after_manifest_fingerprinting(self):
        def modify():
            (self.root/'private-name.png').write_bytes(b'changed')
            (self.root/'other.jpg').write_bytes(b'changed')
        self.on_request=modify
        summary=run_load(self.manifest,self.root/'changed',base_url=self.base,requests=3,warmup=0)
        self.assertEqual(summary['client_error_counts'],{'input_changed':2})
        self.assertEqual(len(self.seen),1)

    def test_invalid_manifest_object_is_rejected_before_output_or_http(self):
        self.manifest.write_text('[]')
        with self.assertRaises(ValueError):
            run_load(self.manifest,self.root/'bad-manifest',base_url=self.base)
        self.assertFalse((self.root/'bad-manifest').exists())

    def test_nested_response_becomes_one_invalid_response_event(self):
        (self.root/'nested.png').write_bytes(b'nested')
        self.manifest.write_text(json.dumps({'images':['nested.png']}))
        summary=run_load(self.manifest,self.root/'nested',base_url=self.base,requests=1,warmup=0)
        self.assertEqual(summary['client_error_counts'],{'invalid_response':1})
        self.assertEqual(len(self.seen),1)

    def test_redirect_and_oversized_response_are_not_retried_or_copied(self):
        for mode, expected_status, expected_error in (('redirect',302,{}),('large',200,{'response_too_large':1})):
            (self.root/'special.png').write_bytes(mode.encode())
            self.manifest.write_text(json.dumps({'images':['special.png']}))
            output=self.root/mode
            before=len(self.seen)
            # Exercise cap+1 with a small cap: large loopback writes can reset
            # on Windows and test transport instead of this size boundary.
            with patch('dice_viewer.recognition.load.MAX_RESPONSE_BYTES',128):
                summary=run_load(self.manifest,output,base_url=self.base,requests=1,warmup=0)
            self.assertEqual(len(self.seen)-before,1)
            self.assertEqual(summary['http_status_counts'],{str(expected_status):1})
            self.assertEqual(summary['client_error_counts'],expected_error)
            self.assertNotIn('PRIVATE_TOKEN',(output/'attempts.jsonl').read_text())


if __name__ == '__main__':
    unittest.main()
