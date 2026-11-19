"""Persistent fixed GPU assignment, one independent 30-task worker per batch."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from profiling.common import atomic_json, read_json

CODE = Path(__file__).resolve().parent


def gpu_inventory():
    def query(flag):
        text=subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True,timeout=15)
        return [[s.strip() for s in line.split(',')] for line in text.splitlines() if line.strip()]
    busy={row[0] for row in query('--query-compute-apps=gpu_uuid,pid')}
    return {int(i):dict(uuid=u,memory=int(m),util=int(p),idle=u not in busy and int(m)<512 and int(p)<=5)
            for i,u,m,p in query('--query-gpu=index,uuid,memory.used,utilization.gpu')}


def worker_alive(pid, root, batch, uuid):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return args[-4:]==['stories260k.train',str(root),str(batch),uuid]
    except (FileNotFoundError,PermissionError,ProcessLookupError):
        return False


def main(root):
    cfg=read_json(CODE/'config.json')
    manifest=root/'data/manifest.json'
    if not manifest.exists():
        raise RuntimeError('Data preparation has not completed')
    with (root/'dispatcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'queue.json'
        jobs=read_json(path)['jobs'] if path.exists() else [
            dict(batch=b,gpu=g,state='pending',failures=0,attempts=0)
            for b,g in zip(cfg['batches'],cfg['gpu_indices'])]
        children={}
        while True:
            for job in jobs:
                batch=job['batch']
                folder=root/'results'/f'batch_{batch:03d}'
                folder.mkdir(parents=True,exist_ok=True)
                if batch in children and children[batch].poll() is not None:
                    del children[batch]
                status_file=folder/'status.json'
                status=read_json(status_file) if status_file.exists() else {}
                if status.get('phase')=='complete':
                    job['state']='complete'
                    continue
                if job['state']=='failed':
                    continue
                if job.get('pid') and worker_alive(job['pid'],root,batch,job['uuid']):
                    job['state']='running'
                    continue
                # Recover a detached child if the dispatcher was interrupted while publishing.
                if status.get('pid') and worker_alive(status['pid'],root,batch,status.get('gpu_uuid')):
                    job.update(pid=status['pid'],uuid=status['gpu_uuid'],state='running')
                    continue
                if job['state']=='running':
                    job['failures']+=int('GPU shared with PIDs' not in status.get('error',''))
                    job['state']='pending' if job['failures']<3 else 'failed'
                    job['error']=status.get('error','worker exited')
                    job['retry_after']=time.time()+60
                if job['state']!='pending' or time.time()<job.get('retry_after',0):
                    continue
                gpu=gpu_inventory()[job['gpu']]
                if not gpu['idle']:
                    continue
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                         TORCHINDUCTOR_COMPILE_THREADS='1',TORCHINDUCTOR_CACHE_DIR=str(folder/'inductor'),
                         TRITON_CACHE_DIR=str(folder/'triton'),PYTHONUNBUFFERED='1')
                with (folder/'worker.log').open('ab') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m','stories260k.train',str(root),
                                            str(batch),gpu['uuid']],stdin=subprocess.DEVNULL,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
                children[batch]=child
                job.update(pid=child.pid,uuid=gpu['uuid'],state='running',attempts=job['attempts']+1,
                           started_unix=time.time())
                print('LAUNCHED',batch,'GPU',job['gpu'],'PID',child.pid,flush=True)
                atomic_json(path,dict(pid=os.getpid(),updated_unix=time.time(),jobs=jobs))
            atomic_json(path,dict(pid=os.getpid(),updated_unix=time.time(),jobs=jobs))
            if all(j['state'] in ('complete','failed') for j in jobs):
                if any(j['state']=='failed' for j in jobs):
                    raise RuntimeError('One or more batches failed; see queue.json')
                return
            time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    main(p.parse_args().root.resolve())
