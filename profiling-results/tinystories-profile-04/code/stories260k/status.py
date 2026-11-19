"""Print each worker's task, seed, epoch and measured progress."""
import argparse
from pathlib import Path
from profiling.common import read_json

def main(root):
    print('batch GPU state         task/30 seed       epoch/4 sequences')
    for j in read_json(root/'queue.json')['jobs']:
        path=root/'results'/f"batch_{j['batch']:03d}"/'status.json'
        s=read_json(path) if path.exists() else {}
        phase=s.get('phase',j['state'])
        print(f"{j['batch']:5} {j['gpu']:3} {phase:13} {s.get('task','-'):>7} {s.get('seed','-'):>10}"
              f" {s.get('epoch','-'):>7} {s.get('sequences_done','-')}")
        if s.get('error'):print(s['error'])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    main(p.parse_args().root.resolve())
