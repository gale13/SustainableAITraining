"""Run paired-seed tasks sequentially on one physical GPU, with atomic recovery."""
import argparse
import fcntl
import hashlib
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from profiling.common import atomic_json, digest, read_json, file_sha256, write_csv
from profiling.energy import Meter, Sensor
from .upstream.model import ModelArgs, Transformer

CODE = Path(__file__).resolve().parent


class ExclusiveSensor(Sensor):
    def __init__(self, uuid, folder):
        super().__init__(uuid)
        self.folder = folder
        if self.nvml is None:
            raise RuntimeError('NVML required for exclusive profiling')

    def read(self):
        others = [p.pid for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
                  if p.pid != os.getpid()]
        if others:
            atomic_json(self.folder/f'overlap_{time.time_ns()}.json',
                        dict(gpu_uuid=self.uuid,pids=others,unix=time.time()))
            raise RuntimeError(f'GPU shared with PIDs {others}')
        return super().read()


def weights_hash(model):
    h = hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()+b'\0')
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fresh_model(seed, cfg):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return Transformer(ModelArgs(**cfg['model']))


def learning_rate(sequences, cfg, total):
    if sequences < cfg['warmup_sequences']:
        return cfg['learning_rate']*sequences/cfg['warmup_sequences']
    fraction = min(1., (sequences-cfg['warmup_sequences'])/(total-cfg['warmup_sequences']))
    return cfg['min_lr'] + .5*(cfg['learning_rate']-cfg['min_lr'])*(1+math.cos(math.pi*fraction))


def checkpoint(path, raw, optimizer, state, signature):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(dict(signature=signature,model=raw.state_dict(),optimizer=optimizer.state_dict(),
                        state=state,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def load_checkpoint(path, raw, optimizer, signature):
    cp = torch.load(path,map_location='cpu',weights_only=True)
    if cp['signature'] != signature:
        raise ValueError('Checkpoint configuration mismatch')
    raw.load_state_dict(cp['model'])
    optimizer.load_state_dict(cp['optimizer'])
    torch.set_rng_state(cp['cpu_rng'])
    torch.cuda.set_rng_state(cp['cuda_rng'])
    return cp['state']


def empty_partial():
    return dict(cursor=0,steps=0,loss_sum=0.,seconds=0.,energy_j=0.,power_integral_j=0.,
                sample_count=0,segments=0,methods=[])


def packed_on_gpu(path, info):
    if path.stat().st_size != info['bytes']:
        raise ValueError('Packed data size mismatch')
    mapped = np.memmap(path,dtype='<i2',mode='r').reshape(info['sequences'],513)
    return torch.from_numpy(np.array(mapped)).to('cuda')


def batches(ids, order, size, start=0):
    for offset in range(start,len(order),size):
        chosen = order[offset:offset+size]
        row = ids[chosen].long()
        yield offset, row[:,:-1].contiguous(), row[:,1:].contiguous()


def loss_forward(model, raw, x, y):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        model(x,y)
        return raw.last_loss


@torch.no_grad()
def evaluate(model, raw, ids, order, meter, phase, epoch, batch):
    model.eval()
    torch.cuda.synchronize()
    meter.begin(phase,block=meter.block,epoch=epoch)
    total = torch.zeros((),dtype=torch.float64,device='cuda')
    for _,x,y in batches(ids,order,batch):
        total += loss_forward(model,raw,x,y).detach().double()*len(x)
    torch.cuda.synchronize()
    measured = meter.end()
    value = total.item()/len(order)
    if not math.isfinite(value):
        raise RuntimeError('Non-finite evaluation loss')
    return value,measured


def run(root, batch, uuid, folder):
    cfg = read_json(CODE/'config.json')
    manifest = read_json(root/'data/manifest.json')
    if batch not in cfg['batches']:
        raise ValueError('Unknown batch size')
    torch.set_num_threads(cfg['threads'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count()!=1 or str(device.uuid).lower().removeprefix('gpu-')!=uuid.lower().removeprefix('gpu-'):
        raise RuntimeError('Training GPU and energy sensor differ')
    signature = digest(dict(config=cfg,data=manifest,batch=batch,
                            model_sha256=file_sha256(CODE/'upstream/model.py'),
                            worker_sha256=file_sha256(Path(__file__))))
    runfile = folder/'run.json'
    if runfile.exists() and read_json(runfile)['signature']!=signature:
        raise ValueError('Experiment settings changed')
    atomic_json(runfile,dict(signature=signature,config=cfg,manifest=manifest,batch=batch,
                            gpu_uuid=uuid,pid=os.getpid(),torch_version=str(torch.__version__)))

    def status(phase, **fields):
        atomic_json(folder/'status.json',dict(phase=phase,batch=batch,gpu_uuid=uuid,
                    pid=os.getpid(),updated_unix=time.time(),**fields))

    completed = [t for t in range(1,cfg['tasks']+1)
                 if (folder/f'task_{t:03d}'/'complete.json').exists()]
    if len(completed)==cfg['tasks']:
        status('complete',completed_tasks=len(completed))
        return
    first = min(set(range(1,cfg['tasks']+1))-set(completed))
    sensor = ExclusiveSensor(uuid,folder)
    meter = Meter(sensor,folder/f'power_{time.time_ns()}.csv.gz',cfg['telemetry_seconds'])
    try:
        status('loading',task=first,completed_tasks=len(completed))
        train = packed_on_gpu(root/'data/train.u16',manifest['train'])
        valid = packed_on_gpu(root/'data/valid.u16',manifest['valid'])
        n = len(train)
        valid_order = torch.arange(len(valid),device='cuda')
        probe_order = torch.as_tensor(np.random.default_rng(cfg['probe_seed']).choice(
            n,cfg['initial_train_probe_sequences'],replace=False),device='cuda')
        original = fresh_model(cfg['seeds'][first-1],cfg)
        raw = original.cuda()
        model = torch.compile(raw) if cfg['compile'] else raw
        status('compiling',task=first,completed_tasks=len(completed))
        meter.label('compile_warmup',first,0)
        # Exercise the full and final partial training batches outside measured epochs.
        warm_optimizer = raw.configure_optimizers(cfg['weight_decay'],cfg['learning_rate'],tuple(cfg['betas']),'cuda')
        model.train()
        for size in sorted({batch,n%batch} - {0}):
            row=train[:size].long()
            loss_forward(model,raw,row[:,:-1].contiguous(),row[:,1:].contiguous()).backward()
            warm_optimizer.step()
            warm_optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            for size in sorted({cfg['eval_batch'],len(valid)%cfg['eval_batch']} - {0}):
                row=valid[:size].long()
                loss_forward(model,raw,row[:,:-1].contiguous(),row[:,1:].contiguous())
        del warm_optimizer
        torch.cuda.synchronize()
        atomic_json(folder/'compile.json',dict(finished_unix=time.time(),parameters=sum(p.numel() for p in raw.parameters()),
                                               train_shapes=sorted({batch,n%batch}-{0})))

        for task in range(first,cfg['tasks']+1):
            target = folder/f'task_{task:03d}'
            target.mkdir(exist_ok=True)
            if (target/'complete.json').exists():
                continue
            seed = cfg['seeds'][task-1]
            started = time.time()
            initial = fresh_model(seed,cfg)
            initial_hash = weights_hash(initial)
            expected_file = root/'expected_initializations.json'
            if expected_file.exists():
                expected = read_json(expected_file)[task-1]
                if (expected['task'],expected['seed'],expected['initial_weights_sha256']) != (task,seed,initial_hash):
                    raise ValueError('Paired-task initial weights differ from the recorded reference')
            raw.load_state_dict(initial.state_dict())
            del initial
            optimizer = raw.configure_optimizers(cfg['weight_decay'],cfg['learning_rate'],tuple(cfg['betas']),'cuda')
            task_signature = digest([signature,task,seed])
            meta = dict(task=task,seed=seed,batch=batch,initial_weights_sha256=initial_hash,
                        parameters=sum(p.numel() for p in raw.parameters()),signature=task_signature)
            initial_path = target/'initial.json'
            if initial_path.exists() and read_json(initial_path)!=meta:
                raise ValueError('Task initialization changed')
            atomic_json(initial_path,meta)
            cp_path = target/'checkpoint.pt'
            state = dict(task=task,epoch=1,global_step=0,sequences=0,history=[],partial=empty_partial())
            if cp_path.exists():
                state = load_checkpoint(cp_path,raw,optimizer,task_signature)
            checkpoint(cp_path,raw,optimizer,state,task_signature)
            if not (target/'baseline.json').exists():
                status('initial_evaluation',task=task,seed=seed,completed_tasks=task-1)
                meter.label('initialization',task,0)
                probe,pm=evaluate(model,raw,train,probe_order,meter,'initial_train_probe',0,cfg['eval_batch'])
                validation,vm=evaluate(model,raw,valid,valid_order,meter,'initial_validation',0,cfg['eval_batch'])
                atomic_json(target/'baseline.json',dict(meta,train_probe_loss=probe,validation_loss=validation,
                            train_probe_sequences=len(probe_order),train_probe_measurement=pm,validation_measurement=vm))
            if state['history']:
                write_csv(target/'epochs.csv',state['history'])
            while state['epoch']<=cfg['epochs']:
                epoch = state['epoch']
                part = state['partial']
                indices = np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(n)
                order = torch.as_tensor(indices,device='cuda')
                order_hash = hashlib.sha256(indices.tobytes()).hexdigest()
                model.train()
                loss_sum = torch.tensor(part['loss_sum'],device='cuda',dtype=torch.float64)
                meter.label('train',task,epoch)
                torch.cuda.synchronize()
                if part['cursor'] < n:
                    meter.begin('train',task,epoch)
                report_at = saved_at = time.monotonic()
                for offset,x,y in batches(train,order,batch,part['cursor']):
                    lr = learning_rate(state['sequences'],cfg,cfg['epochs']*n)
                    for group in optimizer.param_groups:
                        group['lr']=lr
                    loss = loss_forward(model,raw,x,y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['grad_clip'])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    loss_sum += loss.detach().double()*len(x)
                    part['cursor']=offset+len(x)
                    part['steps']+=1
                    state['global_step']+=1
                    state['sequences']+=len(x)
                    now=time.monotonic()
                    if now-report_at>=30 or part['steps']==1:
                        torch.cuda.synchronize()
                        current=loss_sum.item()/part['cursor']
                        if not math.isfinite(current):
                            raise RuntimeError('Non-finite training loss')
                        live=meter.snapshot()
                        status('training',task=task,seed=seed,epoch=epoch,completed_tasks=task-1,
                               sequences_done=part['cursor'],sequences_total=n,train_loss=current,
                               train_seconds=part['seconds']+live['seconds'],
                               train_energy_j=part['energy_j']+live['energy_j'])
                        report_at=now
                    if now-saved_at>=cfg['checkpoint_seconds'] or part['cursor']==n:
                        torch.cuda.synchronize()
                        measured=meter.end()
                        for key in ('seconds','energy_j','power_integral_j','sample_count'):
                            part[key]+=measured[key]
                        if measured['energy_method'] not in part['methods']:
                            part['methods'].append(measured['energy_method'])
                        part['segments']+=1
                        part['loss_sum']=loss_sum.item()
                        if not math.isfinite(part['loss_sum']):
                            raise RuntimeError('Non-finite epoch accumulator')
                        meter.label('checkpoint',task,epoch)
                        checkpoint(cp_path,raw,optimizer,state,task_signature)
                        saved_at=time.monotonic()
                        if part['cursor']<n:
                            meter.begin('train',task,epoch)
                if part['steps']!=math.ceil(n/batch):
                    raise RuntimeError('Epoch update count mismatch')
                status('validation',task=task,seed=seed,epoch=epoch,completed_tasks=task-1)
                validation,vm=evaluate(model,raw,valid,valid_order,meter,'validation',epoch,cfg['eval_batch'])
                probe,pm=evaluate(model,raw,train,probe_order,meter,'train_probe',epoch,cfg['eval_batch'])
                row=dict(task=task,seed=seed,batch=batch,epoch=epoch,
                         initial_weights_sha256=initial_hash,order_sha256=order_hash,
                         train_loss=part['loss_sum']/n,train_probe_loss=probe,validation_loss=validation,
                         train_sequences=n,train_target_tokens=n*512,optimizer_steps=part['steps'],
                         global_step=state['global_step'],learning_rate_end=optimizer.param_groups[0]['lr'],
                         train_seconds=part['seconds'],train_energy_j=part['energy_j'],
                         train_energy_kwh=part['energy_j']/3.6e6,
                         train_mean_power_w=part['energy_j']/part['seconds'],
                         train_power_integral_j=part['power_integral_j'],train_sample_count=part['sample_count'],
                         train_segments=part['segments'],train_energy_method='+'.join(part['methods']),
                         validation_sequences=len(valid),validation_energy_method=vm['energy_method'],
                         probe_energy_method=pm['energy_method'],
                         validation_seconds=vm['seconds'],validation_energy_j=vm['energy_j'],
                         probe_seconds=pm['seconds'],probe_energy_j=pm['energy_j'])
                row.update(epoch_seconds=row['train_seconds']+vm['seconds']+pm['seconds'],
                           epoch_energy_j=row['train_energy_j']+vm['energy_j']+pm['energy_j'])
                state['history'].append(row)
                state['epoch']+=1
                state['partial']=empty_partial()
                meter.label('checkpoint',task,epoch)
                checkpoint(cp_path,raw,optimizer,state,task_signature)
                write_csv(target/'epochs.csv',state['history'])
                print(f"task={task} seed={seed} epoch={epoch} train={row['train_loss']:.6f} "
                      f"valid={validation:.6f} seconds={row['train_seconds']:.2f} joules={row['train_energy_j']:.1f}",flush=True)
            atomic_json(target/'complete.json',dict(meta,epochs=cfg['epochs'],finished_unix=time.time(),
                        final_weights_sha256=weights_hash(raw),latest_attempt_wall_seconds=time.time()-started))
            all_rows=[]
            for task_dir in sorted(folder.glob('task_*')):
                state_file=task_dir/'epochs.csv'
                if state_file.exists():
                    import csv
                    with state_file.open() as stream:
                        all_rows.extend(csv.DictReader(stream))
            write_csv(folder/'epochs.csv',all_rows)
            status('task_complete',task=task,seed=seed,completed_tasks=task)
            del optimizer
        status('complete',completed_tasks=cfg['tasks'])
    finally:
        meter.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('batch',type=int)
    parser.add_argument('uuid')
    args=parser.parse_args()
    root=args.root.resolve()
    folder=root/'results'/f'batch_{args.batch:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            run(root,args.batch,args.uuid,folder)
        except Exception as exc:
            atomic_json(folder/'status.json',dict(phase='failed',pid=os.getpid(),batch=args.batch,
                        gpu_uuid=args.uuid,updated_unix=time.time(),error=repr(exc)))
            traceback.print_exc()
            raise


if __name__=='__main__':
    main()
