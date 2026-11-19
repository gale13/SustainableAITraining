import json,os,sys,shutil,tempfile,time
from pathlib import Path
root=Path('/path/to/experiments/tinystories-profile-04')
sys.path.insert(0,str(root/'code'))
from profiling.common import atomic_json,read_json,file_sha256
from stories260k import train
import numpy as np
import torch
target=Path(tempfile.mkdtemp(prefix='profile04-test-'))
configdir=target/'code'
configdir.mkdir()
shutil.copytree(root/'code/stories260k/upstream',configdir/'upstream')
cfg=read_json(root/'code/stories260k/config.json')
cfg.update(tasks=2,seeds=[1337,1338],epochs=2,warmup_sequences=100,initial_train_probe_sequences=128)
atomic_json(configdir/'config.json',cfg)
train.CODE=configdir
data=target/'data'
data.mkdir()
source=Path('/path/to/experiments/stories260k-smoke.uMUVU05u/data/tok512/01.bin')
tokens=np.memmap(source,mode='r',dtype='<u2')
manifest={}
for name,n in [('train',519),('valid',131)]:
 windows=tokens[np.arange(n)[:,None]*512+np.arange(513)]
 path=data/(name+'.u16')
 path.write_bytes(windows.tobytes())
 manifest[name]=dict(sequences=n,bytes=path.stat().st_size,sha256=file_sha256(path))
atomic_json(data/'manifest.json',manifest)
folder=target/'results'/'batch_512';folder.mkdir(parents=True)
save=train.checkpoint
class CheckpointInterruption(Exception):pass
def interrupted(path,raw,optimizer,state,signature):
 save(path,raw,optimizer,state,signature)
 if state['epoch']==1 and state['partial']['cursor']==519:
  raise CheckpointInterruption()
train.checkpoint=interrupted
try:
 train.run(target,512,os.environ['CUDA_VISIBLE_DEVICES'],folder)
except CheckpointInterruption:
 print('Expected interruption immediately after training checkpoint',flush=True)
else: raise AssertionError('Checkpoint interruption was not exercised')
train.checkpoint=save
train.run(target,512,os.environ['CUDA_VISIBLE_DEVICES'],folder)
assert read_json(folder/'status.json')['phase']=='complete'
for task in (1,2):
 import csv
 directory=folder/f'task_{task:03d}'
 with (directory/'epochs.csv').open() as f: rows=list(csv.DictReader(f))
 assert len(rows)==2
 assert all(int(r['optimizer_steps'])==2 and int(r['train_sequences'])==519 for r in rows)
 assert int(rows[1]['global_step'])==4
 assert float(rows[0]['train_energy_j'])>0
 assert int(rows[1]['seed'])==1336+task
a=read_json(folder/'task_001/initial.json');b=read_json(folder/'task_002/initial.json')
assert a['initial_weights_sha256']!=b['initial_weights_sha256']
assert a['parameters']==260032
print('PASS full worker, batch512, partial batches, 2 fresh tasks, checkpoints, resumed full training epoch, energy, seeds',flush=True)
print('ARTIFACT',target,flush=True)
