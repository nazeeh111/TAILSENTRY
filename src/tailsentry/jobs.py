"""One owned estimation process and its explicit, bounded lifecycle."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid


class EstimateSession:
    def __init__(self, records, deadline=120):
        self.records = records
        self.deadline = deadline
        self.lock = threading.Lock()
        self.active = None
        self.status = {'state': 'idle', 'job_id': None}
        self.last_record = None

    def start(self, config):
        payload = json.dumps(config, allow_nan=False).encode()
        if len(payload) > 8192:
            raise ValueError('Request too large')
        with self.lock:
            if self.active is not None:
                raise RuntimeError('An estimate is already running')
            output = tempfile.TemporaryFile()
            error = tempfile.TemporaryFile()
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            try:
                process = subprocess.Popen([sys.executable, '-m', 'tailsentry.worker'],
                    stdin=subprocess.PIPE, stdout=output, stderr=error, env=env, close_fds=True)
                process.stdin.write(payload)
                process.stdin.close()
            except BaseException:
                if 'process' in locals():
                    process.kill()
                    process.wait()
                output.close(); error.close()
                raise
            job = {'id': uuid.uuid4().hex, 'process': process, 'output': output,
                   'error': error, 'cancel': False, 'start': time.monotonic()}
            self.active = job
            self.status = {'state': 'running', 'job_id': job['id']}
            threading.Thread(target=self._monitor, args=(job,), daemon=True).start()
            return dict(self.status)

    def cancel(self, job_id):
        with self.lock:
            if self.active is None:
                if self.status['job_id'] != job_id:
                    raise RuntimeError('That job is no longer active')
                return dict(self.status)
            if self.active['id'] != job_id:
                raise RuntimeError('That job is no longer active')
            self.active['cancel'] = True
            self.status = {'state': 'cancelling', 'job_id': self.active['id']}
            return dict(self.status)

    def snapshot(self):
        with self.lock:
            return {**self.status, 'last_record': self.last_record}

    def _monitor(self, job):
        process = job['process']
        terminal = None
        try:
            while process.poll() is None:
                with self.lock:
                    cancelled = job['cancel']
                if cancelled or time.monotonic() - job['start'] >= self.deadline:
                    terminal = 'cancelled' if cancelled else 'timed-out'
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    break
                time.sleep(.025)
            process.wait()
            if terminal is None and time.monotonic() - job['start'] >= self.deadline:
                terminal = 'timed-out'
            job['output'].seek(0)
            raw = job['output'].read(4 * 1024 * 1024 + 1)
            if terminal is None:
                if len(raw) > 4 * 1024 * 1024:
                    raise ValueError('Worker result exceeded size limit')
                result = json.loads(raw)
                if process.returncode != 0 or 'error' in result:
                    raise ValueError(result.get('error', 'Estimate worker failed'))
                record = result['record']
            with self.lock:
                # Cancellation ordered before adoption always wins, including a
                # child that exited just before the cancellation request.
                if job['cancel']:
                    terminal = 'cancelled'
                if terminal is None:
                    self.records.add(record)
                    self.last_record = record
                    terminal = 'insufficient-budget' if record.get('status') == 'insufficient-budget' else 'completed'
                self.status = {'state': terminal, 'job_id': job['id']}
                self.active = None
        except Exception as error:
            with self.lock:
                self.status = {'state': 'cancelled' if job['cancel'] else 'failed',
                               'job_id': job['id'], 'error': str(error)[:1000]}
                self.active = None
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
            job['output'].close(); job['error'].close()

    def shutdown(self):
        with self.lock:
            job = self.active
            if job:
                job['cancel'] = True
        if job:
            job['process'].wait(timeout=self.deadline + 3)
