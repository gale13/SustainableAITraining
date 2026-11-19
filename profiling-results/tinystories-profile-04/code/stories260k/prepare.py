"""Pack the exact 50 upstream shards with the published 512-piece tokenizer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np
from sentencepiece import SentencePieceProcessor
from profiling.common import atomic_json, file_sha256, read_json

CODE = Path(__file__).resolve().parent


def tokenize_shard(args):
    source, directory, expected = args
    tokenizer = SentencePieceProcessor(model_file=str(CODE/'tok512.model'))
    stories = read_json(source)
    path = directory/(source.stem+'.u16')
    total = 0
    with path.with_suffix('.tokens').open('wb') as stream:
        for start in range(0,len(stories),512):
            tokens = tokenizer.encode([r['story'].strip() for r in stories[start:start+512]],
                                      out_type=int,add_bos=True,add_eos=False,num_threads=2)
            values = np.concatenate([np.asarray(row,dtype='<u2') for row in tokens])
            stream.write(values.tobytes())
            total += len(values)
    assert total==expected['raw_tokens'] and len(stories)==expected['stories']
    count = total//512-1
    assert count==expected['usable_windows']
    tokens = np.memmap(path.with_suffix('.tokens'),mode='r',dtype='<u2')
    with path.open('wb') as stream:
        for start in range(0,count,4096):
            indices = np.arange(start,min(start+4096,count))[:,None]*512+np.arange(513)
            stream.write(tokens[indices].tobytes())
    del tokens
    path.with_suffix('.tokens').unlink()
    source.unlink()
    return dict(name=path.name,sequences=count,bytes=path.stat().st_size,sha256=file_sha256(path))


def main(root):
    data = root/'data'
    data.mkdir(parents=True,exist_ok=True)
    with (data/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (data/'manifest.json').exists():
            print('Prepared data already available',flush=True)
            return
        reference = read_json(CODE/'epoch_count.json')
        assert file_sha256(CODE/'tok512.model')==reference['tokenizer_sha256']
        archive=data/'TinyStories_all_data.tar.gz'
        if not archive.exists():
            # Cache the full corpus locally for reproducible later runs.
            for domain in ('huggingface.co','hf-mirror.com'):
                url=f'https://{domain}/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz'
                partial=archive.with_suffix('.download')
                if shutil.which('aria2c'):
                    command=['aria2c','--continue=true','--file-allocation=none',
                             '--auto-file-renaming=false','--allow-overwrite=true',
                             '--max-connection-per-server=8','--split=8','--min-split-size=4M',
                             '--max-tries=3','--retry-wait=5','--console-log-level=warn',
                             '--checksum=sha-256='+reference['archive_sha256'],
                             '--dir='+str(data),'--out='+partial.name,url]
                else:
                    command=['curl','-fL','--retry','2','--connect-timeout','20',
                             '--speed-time','60','--speed-limit','1024','-C','-','-o',str(partial),url]
                result=subprocess.run(command)
                if result.returncode==0:
                    archive.with_suffix('.download').replace(archive)
                    break
            else:
                raise RuntimeError('Corpus download failed')
        assert file_sha256(archive)==reference['archive_sha256'], 'Corpus checksum mismatch'
        expected={x['shard']:x for x in reference['shards']}
        staging=data/'packing'
        staging.mkdir(exist_ok=True)
        pending=[]
        with ProcessPoolExecutor(max_workers=2) as pool,tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                name=Path(member.name).name
                if name not in expected or not member.isfile():
                    continue
                path=staging/name
                with tar.extractfile(member) as src,path.open('wb') as dst:
                    shutil.copyfileobj(src,dst,length=2**20)
                pending.append(pool.submit(tokenize_shard,(path,staging,expected[name])))
                if len(pending)>=4:
                    record=pending.pop(0).result()
                    print('Packed',record['name'],flush=True)
            for future in pending:
                record=future.result()
                print('Packed',record['name'],flush=True)
        records=[]
        for split,names in [('valid',['data00.u16']),('train',[f'data{i:02d}.u16' for i in range(1,50)])]:
            output=data/(split+'.u16')
            with output.with_suffix('.tmp').open('wb') as dst:
                for name in names:
                    with (staging/name).open('rb') as src:
                        shutil.copyfileobj(src,dst,length=4*2**20)
            output.with_suffix('.tmp').replace(output)
            records.append((split,dict(sequences=output.stat().st_size//(513*2),
                                       sha256=file_sha256(output),bytes=output.stat().st_size)))
        manifest=dict(records,source_archive_sha256=reference['archive_sha256'],
                      tokenizer_sha256=reference['tokenizer_sha256'],sequence_length=512,
                      layout='uint16 rows of 513 tokens, x=row[:-1], y=row[1:]',
                      train_shards=list(range(1,50)),validation_shard=0,
                      official_equivalent_epochs=reference['equivalent_epochs'])
        assert manifest['train']['sequences']==3541812
        atomic_json(data/'manifest.json',manifest)
        print('READY',json.dumps(manifest),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root.resolve())
