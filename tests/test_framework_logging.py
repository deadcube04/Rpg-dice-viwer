"""The observer must surround the actual Bento timeout/admission layers."""

import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import bentoml
from starlette.responses import JSONResponse

from dice_viewer.recognition.inputs import BinaryImage
from dice_viewer.recognition.http_service import ObservedService


async def call(app):
    messages=[]
    async def receive():
        return {'type':'http.request','body':b'invalid','more_body':False}
    async def send(message):
        messages.append(message)
    await app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1',
               'scheme':'http','method':'POST','path':'/v1/recognize',
               'raw_path':b'/v1/recognize','query_string':b'',
               'root_path':'','headers':[(b'content-type',b'image/png')],
               'client':('127.0.0.1',40000),'server':('127.0.0.1',25000)},receive,send)
    return messages[0]['status'],b''.join(m.get('body',b'') for m in messages[1:])


class FrameworkLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_admission_429_is_logged_once_without_changing_response(self):
        entered,release=asyncio.Event(),asyncio.Event()

        @bentoml.service(name='AdmissionLoggingTest',service_class=ObservedService,
                         traffic={'max_concurrency':1,'timeout':5},metrics={'enabled':False})
        class Stub:
            @bentoml.api(route='/v1/recognize',input_spec=BinaryImage)
            async def recognize(self,image:BinaryImage,/):
                entered.set()
                await release.wait()
                return JSONResponse({'ok':True})

        with TemporaryDirectory() as directory,patch.dict('os.environ',{'DICE_OBSERVABILITY_DIR':directory}):
            app=Stub.to_asgi()
            async with app.app.router.lifespan_context(app.app):
                first=asyncio.create_task(call(app))
                try:
                    await asyncio.wait_for(entered.wait(),5)
                    status,body=await call(app)
                    self.assertEqual(status,429)
                    self.assertEqual(json.loads(body),{'error':'Too many requests'})
                finally:
                    release.set()
                    await first
            rows=[json.loads(line) for line in (Path(directory)/'recognition.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            rejected,=[r for r in rows if r['http_status']==429]
            self.assertEqual(rejected['error_code'],'framework_rejected')
            self.assertEqual(rejected['http_observation_version'],2)
            self.assertTrue(rejected['response_completed'])
            self.assertEqual(rejected['input']['received_bytes'],0)

    async def test_native_timeout_504_is_logged_as_the_actual_response(self):
        @bentoml.service(name='TimeoutLoggingTest',service_class=ObservedService,
                         traffic={'timeout':0.02},metrics={'enabled':False})
        class Stub:
            @bentoml.api(route='/v1/recognize',input_spec=BinaryImage)
            async def recognize(self,image:BinaryImage,/):
                await asyncio.sleep(1)
                return JSONResponse({'ok':True})

        with TemporaryDirectory() as directory,patch.dict('os.environ',{'DICE_OBSERVABILITY_DIR':directory}):
            app=Stub.to_asgi()
            async with app.app.router.lifespan_context(app.app):
                status,body=await call(app)
                self.assertEqual(status,504)
                self.assertIn('error',json.loads(body))
                await asyncio.sleep(0)
            rows=[json.loads(line) for line in (Path(directory)/'recognition.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['http_status'],504)
            self.assertEqual(rows[0]['error_code'],'framework_timeout')
            self.assertTrue(rows[0]['response_completed'])

    async def test_framework_500_has_actual_status_without_exception_text_in_jsonl(self):
        @bentoml.service(name='ErrorLoggingTest',service_class=ObservedService,
                         metrics={'enabled':False})
        class Stub:
            @bentoml.api(route='/v1/recognize',input_spec=BinaryImage)
            async def recognize(self,image:BinaryImage,/):
                raise RuntimeError('controlled_failure')

        with TemporaryDirectory() as directory,patch.dict('os.environ',{'DICE_OBSERVABILITY_DIR':directory}):
            app=Stub.to_asgi()
            async with app.app.router.lifespan_context(app.app):
                with self.assertLogs('bentoml',level='ERROR'):
                    status,_=await call(app)
                self.assertEqual(status,500)
            text=(Path(directory)/'recognition.jsonl').read_text()
            rows=[json.loads(line) for line in text.splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['http_status'],500)
            self.assertEqual(rows[0]['error_code'],'framework_error')
            self.assertNotIn('controlled_failure',text)


if __name__=='__main__':
    unittest.main()
