import argparse,hashlib,json
from pathlib import Path
a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--upper-root',type=Path,required=True);a.add_argument('--lift',type=Path,required=True);o=a.parse_args();m=json.loads((Path(__file__).parent/'manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(o.source)==m['source']['sha256'];assert sha(o.lift)==m['lift_sha256'];ls=o.source.read_text().splitlines()
for e in m['entries']:assert 1<=e['line_start']<e['line_end_exclusive']<=len(ls)+1
for n,h in m['formal_sources'].items():assert sha(o.upper_root/n)==h,n
print('PASS: exact source/module identities and 12 locator ranges; no mathematical proof/replay performed')
