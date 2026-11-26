"""Verify CLI publication and retry at the filesystem boundary."""

import json
import os
import signal
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest


@pytest.mark.skipif(os.name != 'posix', reason='requires a POSIX file size limit')
def test_failed_estimate_write_can_retry_and_preserves_existing_file(tmp_path):
    def limit_file_size():
        import resource

        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024, 1024))

    output = tmp_path / 'estimate.json'
    command = [sys.executable, '-m', 'tailsentry.cli', 'estimate', '--budget', '4096',
               '--seed', '42', '--output', str(output)]
    failure = subprocess.run(command, capture_output=True, text=True, timeout=30,
                             preexec_fn=limit_file_size)
    assert failure.returncode == 1
    assert 'Could not save estimate' in failure.stderr
    assert 'Traceback' not in failure.stderr
    assert not output.exists()
    assert not list(tmp_path.glob('.tailsentry-*'))

    subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    original = output.read_bytes()
    assert json.loads(original)['record_sha256']

    conflict = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert conflict.returncode == 1
    assert 'Output already exists' in conflict.stderr
    assert output.read_bytes() == original


def test_estimate_reports_unusable_parent_without_calling_it_a_destination_conflict(tmp_path):
    parent = tmp_path / 'file'
    parent.write_text('original')
    output = parent / 'estimate.json'
    command = [sys.executable, '-m', 'tailsentry.cli', 'estimate', '--budget', '4096',
               '--seed', '42', '--output', str(output)]
    failure = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert failure.returncode == 1
    assert 'Could not save estimate' in failure.stderr
    assert 'Output already exists' not in failure.stderr
    assert parent.read_text() == 'original'


def test_completed_estimate_is_kept_when_cleanup_fails(tmp_path, capsys):
    from tailsentry.cli import _save_estimate

    output = tmp_path / 'estimate.json'
    with patch.object(tempfile.TemporaryDirectory, 'cleanup',
                      side_effect=OSError('controlled cleanup failure')):
        _save_estimate(output, '{"estimate":0.001}')
    assert json.loads(output.read_text()) == {'estimate': 0.001}
    message = capsys.readouterr().err
    assert 'Estimate saved' in message
    assert 'could not clean staging directory' in message
