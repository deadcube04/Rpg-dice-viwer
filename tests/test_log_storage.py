"""Rotation/expiration only operate on this logger's closed segments."""

from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dice_viewer.recognition.log_storage import AppendOnlyLog, LogPolicy
from dice_viewer.recognition.observability import JsonlWriter


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "recognition.jsonl"
        self.now = datetime(2026, 11, 17, 18, tzinfo=timezone.utc)

    def storage(self, **policy):
        return AppendOnlyLog(self.path, LogPolicy(**policy), clock=lambda: self.now)

    def append(self, store, payload):
        store.append(payload)
        # Emulate OS mtime using the controlled clock, for daily rotation tests.
        os.utime(self.path, (self.now.timestamp(), self.now.timestamp()))

    def archives(self):
        return list(self.root.glob("recognition.*.jsonl"))

    def test_size_rotation_preserves_closed_bytes_and_keeps_batch_together(self):
        store = self.storage(max_bytes=20, max_total_bytes=200)
        first = b'{"n":1}\n{"n":2}\n'
        self.append(store, first)
        self.append(store, b'{"n":3}\n')
        archive, = self.archives()
        self.assertEqual(archive.read_bytes(), first)
        self.assertEqual(self.path.read_bytes(), b'{"n":3}\n')
        self.append(store, b'{"n":4}\n')
        self.assertEqual(archive.read_bytes(), first)

    def test_utc_day_rotation_and_restart_keep_previous_records(self):
        self.append(self.storage(), b'first\n')
        self.now += timedelta(days=1)
        self.append(self.storage(), b'second\n')
        archive, = self.archives()
        self.assertEqual(archive.read_bytes(), b'first\n')
        self.assertEqual(self.path.read_bytes(), b'second\n')

    def test_expiration_removes_only_owned_closed_segments(self):
        store = self.storage(max_bytes=10, max_total_bytes=1000, retention_days=2)
        self.append(store, b'old-data\n')
        self.now += timedelta(days=3)
        unrelated = self.root / 'someone-else.jsonl'
        unrelated.write_bytes(b'keep')
        self.append(store, b'new-data\n')
        self.assertEqual(self.archives(), [])
        self.assertEqual(self.path.read_bytes(), b'new-data\n')
        self.assertEqual(unrelated.read_bytes(), b'keep')

    def test_total_cap_removes_oldest_archives_but_never_active(self):
        store = self.storage(max_bytes=10, max_total_bytes=20)
        for value in (b'111111111\n', b'222222222\n', b'333333333\n'):
            self.append(store, value)
            self.now += timedelta(seconds=1)
        self.assertEqual(self.path.read_bytes(), b'333333333\n')
        archive, = self.archives()
        self.assertEqual(archive.read_bytes(), b'222222222\n')

    def test_oversized_batch_is_rejected_without_changing_existing_bytes(self):
        store = self.storage(max_bytes=10, max_total_bytes=20)
        self.append(store, b'keep\n')
        with self.assertRaises(ValueError):
            store.append(b'x' * 21)
        self.assertEqual(self.path.read_bytes(), b'keep\n')
        self.assertEqual(self.archives(), [])

    def test_fsync_is_called_and_failure_does_not_retry_batch(self):
        store = self.storage()
        with patch('dice_viewer.recognition.log_storage.os.fsync', side_effect=OSError('private')) as sync:
            with self.assertRaises(OSError):
                store.append(b'one\n')
        self.assertEqual(sync.call_count, 1)
        self.assertEqual(self.path.read_bytes(), b'one\n')

    def test_foreign_similar_name_is_not_expired(self):
        foreign = self.root / 'recognition.not-a-managed-segment.jsonl'
        foreign.write_bytes(b'keep')
        self.append(self.storage(), b'active\n')
        self.assertEqual(foreign.read_bytes(), b'keep')

    def test_partial_tail_is_preserved_separately_from_new_complete_line(self):
        self.path.write_bytes(b'{"interrupted":')
        self.append(self.storage(), b'{"complete":true}\n')
        archive, = self.archives()
        self.assertEqual(archive.read_bytes(), b'{"interrupted":')
        self.assertEqual(self.path.read_bytes(), b'{"complete":true}\n')

    def test_cleanup_failure_does_not_retry_or_leak_exception(self):
        store = self.storage()
        with patch.object(store, '_expire', side_effect=OSError('PRIVATE_TOKEN')):
            with self.assertLogs('dice_viewer.recognition.observability', level='ERROR') as diagnostic:
                self.append(store, b'one\n')
        self.assertEqual(self.path.read_bytes(), b'one\n')
        self.assertIn('observability_retention_failed', diagnostic.output[0])
        self.assertNotIn('PRIVATE_TOKEN', str(diagnostic.output))

    def test_closed_symlink_and_its_target_are_not_expired(self):
        target = self.root / 'outside.jsonl'
        target.write_bytes(b'keep')
        link = self.root / ('recognition.20260101T000000.000000Z.' + 'a'*32 + '.jsonl')
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest('Host does not permit creating symlinks')
        self.append(self.storage(), b'active\n')
        self.assertTrue(link.is_symlink())
        self.assertEqual(target.read_bytes(), b'keep')

    def test_invalid_policy_is_rejected(self):
        for settings in ({'max_bytes': 0}, {'retention_days': -1},
                         {'retention_days': 999999999},
                         {'max_bytes': 100, 'max_total_bytes': 50}):
            with self.assertRaises(ValueError):
                LogPolicy(**settings)

    def test_environment_limits_are_applied_to_the_real_writer(self):
        values = {'DICE_OBSERVABILITY_DIR': str(self.root), 'DICE_LOG_MAX_BYTES': '20',
                  'DICE_LOG_RETENTION_DAYS': '7', 'DICE_LOG_MAX_TOTAL_BYTES': '100'}
        with patch.dict(os.environ, values):
            writer = JsonlWriter.from_environment()
        self.assertEqual(writer.storage.policy, LogPolicy(max_bytes=20, retention_days=7, max_total_bytes=100))
        writer.write([{'n': 1}])
        self.assertEqual(self.path.read_bytes(), b'{"n":1}\n')

    def test_invalid_environment_uses_defaults_without_leaking_value(self):
        with patch.dict(os.environ, {'DICE_LOG_RETENTION_DAYS': 'PRIVATE_TOKEN'}):
            with self.assertLogs('dice_viewer.recognition.observability', level='ERROR') as diagnostic:
                writer = JsonlWriter.from_environment()
        self.assertEqual(writer.storage.policy, LogPolicy())
        self.assertIn('observability_config_invalid', str(diagnostic.output))
        self.assertNotIn('PRIVATE_TOKEN', str(diagnostic.output))


if __name__ == '__main__':
    unittest.main()
