import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest
from tailsentry.jobs import EstimateSession
from tailsentry.service import RecordStore, investigate


def wait(session, timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=session.snapshot()
        if value['state'] not in ('running','cancelling'):
            return value
        time.sleep(.02)
    raise AssertionError('job did not finish')


def stub(script):
    real=subprocess.Popen
    def launch(_command, **kwargs):
        return real([sys.executable,'-c',script], **kwargs)
    return launch


def test_hung_timeout_reaps_only_owned_child_then_next_job_succeeds():
    session=EstimateSession(RecordStore(),deadline=.25)
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        with patch('tailsentry.jobs.subprocess.Popen',side_effect=stub('import time;time.sleep(30)')):
            session.start({'case':'single_1e3','budget':256})
            owned=session.active['process']
            assert wait(session)['state']=='timed-out'
        assert owned.poll() is not None and unrelated.poll() is None
        session.deadline=15
        session.start({'case':'single_1e3','budget':256,'seed':7})
        result=wait(session)
        assert result['state'] in ('completed','insufficient-budget')
        assert result['last_record']['estimate']==pytest.approx(.001,rel=1e-12)
    finally:
        session.shutdown();unrelated.terminate();unrelated.wait(timeout=5)


def test_cancel_preserves_completed_record_and_wrong_id_cannot_cancel_next():
    session=EstimateSession(RecordStore(),deadline=15)
    session.start({'case':'single_1e3','budget':256,'seed':7})
    previous=wait(session)['last_record']
    with patch('tailsentry.jobs.subprocess.Popen',side_effect=stub('import time;time.sleep(30)')):
        job=session.start({'case':'single_1e3','budget':256})
        process=session.active['process']
        with pytest.raises(RuntimeError):session.cancel('obsolete')
        session.cancel(job['job_id'])
        status=wait(session)
    assert status['state']=='cancelled' and status['last_record']==previous
    assert process.poll() is not None
    session.start({'case':'single_1e3','budget':256,'seed':8})
    assert wait(session)['last_record']['seed']==8
    session.shutdown()


def test_worker_retains_fixed_seed_and_separate_oracle_accounting():
    config={'case':'single_1e3','method':'envelope_projected_fixed','budget':256,'seed':7}
    direct=investigate(config)
    session=EstimateSession(RecordStore(),deadline=15)
    session.start(config);record=wait(session)['last_record'];session.shutdown()
    for name in ('estimate','interval','point_evaluations','conditional_evaluations','attribution_conditional_calls','status'):
        assert record[name]==direct[name]


def test_slow_and_truncated_body_do_not_hold_estimator_slot():
    listener=socket.socket();listener.bind(('127.0.0.1',0));port=listener.getsockname()[1];listener.close()
    process=subprocess.Popen([sys.executable,'-c',f'from tailsentry.service import serve; serve({port})'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    slow=None
    try:
        deadline=time.monotonic()+10
        while True:
            try:slow=socket.create_connection(('127.0.0.1',port),timeout=.2);break
            except OSError:
                if process.poll() is not None or time.monotonic()>deadline:raise
                time.sleep(.05)
        slow.settimeout(4)
        slow.sendall((f'POST /api/run HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nOrigin: http://127.0.0.1:{port}\r\nContent-Type: application/json\r\nX-Tailsentry: local\r\nContent-Length: 100\r\n\r\n{{').encode())
        payload=json.dumps({'case':'single_1e3','budget':256,'seed':7}).encode()
        req=urllib.request.Request(f'http://127.0.0.1:{port}/api/run',data=payload,headers={'Content-Type':'application/json','X-Tailsentry':'local','Origin':f'http://127.0.0.1:{port}'})
        with urllib.request.urlopen(req,timeout=4) as response:
            assert response.status==202
        assert b' 408 ' in slow.recv(1024)
        truncated=socket.create_connection(('127.0.0.1',port),timeout=2)
        truncated.sendall((f'POST /api/run HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nOrigin: http://127.0.0.1:{port}\r\nContent-Type: application/json\r\nX-Tailsentry: local\r\nContent-Length: 10\r\n\r\n{{').encode())
        truncated.shutdown(socket.SHUT_WR)
        assert b' 400 ' in truncated.recv(1024)
        truncated.close()
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status',timeout=2) as response: status=json.load(response)
            if status['state'] not in ('running','cancelling'):break
            time.sleep(.05)
        assert status['state'] in ('completed','insufficient-budget')
    finally:
        if slow:slow.close()
        process.terminate();process.wait(timeout=5)


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process observation')
def test_server_termination_reaps_its_active_worker():
    listener=socket.socket();listener.bind(('127.0.0.1',0));port=listener.getsockname()[1];listener.close()
    server=subprocess.Popen([sys.executable,'-c',f'from tailsentry.service import serve; serve({port})'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+10
        while True:
            try:socket.create_connection(('127.0.0.1',port),timeout=.2).close();break
            except OSError:
                if server.poll() is not None or time.monotonic()>deadline:raise
                time.sleep(.05)
        origin=f'http://127.0.0.1:{port}'
        req=urllib.request.Request(origin+'/api/run',data=json.dumps({'budget':65536,'seed':7}).encode(),headers={'Content-Type':'application/json','X-Tailsentry':'local','Origin':origin})
        with urllib.request.urlopen(req,timeout=4) as response:assert response.status==202
        deadline=time.monotonic()+5; children=[]
        while not children and time.monotonic()<deadline:
            entries=subprocess.check_output(['ps','-A','-o','pid=,ppid='],text=True).splitlines()
            children=[int(pid) for pid,parent in (entry.split() for entry in entries) if int(parent)==server.pid]
            if not children:time.sleep(.01)
        assert children
        server.terminate();server.wait(timeout=8)
        for child in children:
            with pytest.raises(ProcessLookupError):os.kill(child,0)
    finally:
        if server.poll() is None:server.kill();server.wait()
