"""Opt-in live reconciliation; requires an idle instrumented local API."""

import json
from collections import Counter
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from urllib.request import urlopen

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from prometheus_client.parser import text_string_to_metric_families
from dice_viewer.recognition.load import run_load


@unittest.skipUnless(os.environ.get('DICE_INTEGRATION_URL'), 'Live API integration is opt-in')
class LiveLoadTests(unittest.TestCase):
    def test_attempts_match_jsonl_features_dice_and_native_metrics(self):
        self.check_load(concurrency=1,count=4)

    def test_concurrent_attempts_preserve_rejections_and_native_counts(self):
        self.check_load(concurrency=3,count=6)

    def check_load(self,concurrency,count):
        base=os.environ['DICE_INTEGRATION_URL'].rstrip('/')
        manifest=Path(os.environ['DICE_INTEGRATION_MANIFEST'])
        log=Path(os.environ.get('DICE_INTEGRATION_LOG',
                               'runtime/observability-stage2/state/observability/recognition.jsonl'))

        def counter():
            with urlopen(base+'/metrics',timeout=10) as response:
                samples=[s for f in text_string_to_metric_families(response.read().decode()) for s in f.samples]
            return sum(s.value for s in samples if s.name=='bentoml_service_request_total'
                       and s.labels.get('endpoint')=='/v1/recognize')

        def events():
            result=[]
            for path in log.parent.glob('recognition*.jsonl'):
                for line in path.read_text(encoding='utf-8').splitlines():
                    try:
                        result.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            return result

        before_ids={r['request_id'] for r in events() if r['event']=='request'}
        before=counter()
        with TemporaryDirectory() as directory:
            output=Path(directory)/'run'
            summary=run_load(manifest,output,base_url=base,requests=count,warmup=1,concurrency=concurrency)
            attempts=[json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines()]
        self.assertEqual(len(attempts),count+1)
        self.assertEqual(summary['client_error_counts'],{})
        ids={attempt['request_id'] for attempt in attempts if attempt['request_id'] is not None}
        self.assertEqual(len(ids),sum(attempt['request_id'] is not None for attempt in attempts))
        if concurrency==1:
            self.assertEqual(len(ids),count+1)
        # The client may finish receiving before the server flushes its JSONL.
        found=[]
        for _ in range(100):
            found=[r for r in events() if r.get('request_id') not in before_ids]
            if sum(r['event']=='request' for r in found)>=count+1:
                break
            time.sleep(0.05)
        self.assertEqual(counter()-before,count+1)
        requests=[r for r in found if r['event']=='request']
        self.assertEqual(len(requests),count+1)
        self.assertEqual(Counter(r['http_status'] for r in requests),
                         Counter(r['http_status'] for r in attempts))
        framework=[r for r in requests if r['error_code']=='framework_rejected']
        self.assertEqual(len(framework),sum(r['http_status']==429 and r['request_id'] is None for r in attempts))
        for attempt in attempts:
            self.assertIn(attempt['http_status'],(200,404) if concurrency==1 else (200,404,429,503))
            if attempt['http_status']==429 and attempt['request_id'] is None:
                continue  # Covered by exact status/count matching above; native body has no public ID.
            records=[e for e in found if e['event']=='request' and e['request_id']==attempt['request_id']]
            self.assertEqual(len(records),1)
            event=records[0]
            self.assertEqual(event['http_status'],attempt['http_status'])
            self.assertEqual(event['image_id'],attempt['image_id'])
            self.assertEqual(event['input']['received_bytes'],attempt['input_bytes'])
            if attempt['http_status'] in (200,404):
                self.assertEqual(event['input']['image_features']['status'],'ok')
                self.assertEqual(event['output']['status'],attempt['status'])
            else:
                self.assertEqual(event['error_code'],attempt['error_code'])
            dice=[e for e in found if e['event']=='die' and e['request_id']==attempt['request_id']]
            self.assertEqual(len(dice),event['output'].get('dices_count',0))


if __name__=='__main__':
    unittest.main()
