"""Guard scientific source identity before mounting current instrumentation."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from observe import verify_overlay


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'workspace'
        self.release=Path(self.tmp.name)/'release'
        self.release.mkdir()
        self.source=self.root/'src/dice_viewer/models/frozen.py'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'FROZEN = True\n')
        (self.root/'src/dice_viewer/__init__.py').write_text('"""Package description."""\n')
        manifest={'files':{'src/dice_viewer/models/frozen.py':hashlib.sha256(self.source.read_bytes()).hexdigest()}}
        (self.release/'RELEASE.json').write_text(json.dumps(manifest))

    def test_identical_science_and_new_recognition_instrumentation_are_accepted(self):
        instrument=self.root/'src/dice_viewer/recognition/observability.py'
        instrument.parent.mkdir()
        instrument.write_text('INSTRUMENTATION = True\n')
        verify_overlay(self.root,self.release)

    def test_changed_scientific_module_is_rejected(self):
        self.source.write_text('FROZEN = False\n')
        with self.assertRaises(ValueError):
            verify_overlay(self.root,self.release)

    def test_unapproved_code_in_root_initializer_is_rejected(self):
        (self.root/'src/dice_viewer/__init__.py').write_text('import something\n')
        with self.assertRaises(ValueError):
            verify_overlay(self.root,self.release)

    def test_missing_scientific_module_is_rejected_before_docker_start(self):
        self.source.unlink()
        with self.assertRaises(ValueError):
            verify_overlay(self.root,self.release)


if __name__=='__main__':
    unittest.main()
