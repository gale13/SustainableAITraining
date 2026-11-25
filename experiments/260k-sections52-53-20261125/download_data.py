"""Fetch the full, hash-verified release into this experiment directory."""
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parent


def main():
    info=json.loads((ROOT/'release_data.json').read_text())
    with tempfile.TemporaryDirectory(prefix='260k-section-data-') as temp:
        archive=Path(temp)/'full.zip'
        print('Downloading',info['url'],flush=True)
        urllib.request.urlretrieve(info['url'],archive)
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==info['sha256'],'Archive checksum mismatch'
        with ZipFile(archive) as z:
            assert z.testzip() is None
            for item in z.infolist():
                rel=Path(item.filename)
                assert not rel.is_absolute() and '..' not in rel.parts
                if item.is_dir():continue
                target=ROOT/rel
                with z.open(item) as src:
                    data=src.read()
                if target.exists() and target.read_bytes()!=data:
                    raise SystemExit(f'Local file differs: {rel}. Extract the ZIP into a new directory to preserve your edits.')
            for item in z.infolist():
                if item.is_dir():continue
                target=ROOT/item.filename
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(item) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)
    print('Full data installed. Run python reproduce.py verify.')


if __name__=='__main__':main()
