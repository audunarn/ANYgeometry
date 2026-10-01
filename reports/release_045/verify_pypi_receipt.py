"""Bind public index publication to the exact authorized prebuilt assets."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root/'pypi-raw.json').read_text(encoding='utf-8-sig'))
expected = json.loads((root/'payload-review.json').read_text())['artifacts']
assert data['info']['version'] == '0.4.5'
observed = sorted(({'filename':row['filename'],'bytes':row['size'],'sha256':row['digests']['sha256']} for row in data['urls']), key=lambda row:row['filename'])
assert observed == expected, (observed, expected)
receipt = {'status':'verified','distribution':'ANYgeometry','version':'0.4.5',
           'url':'https://pypi.org/project/ANYgeometry/0.4.5/', 'artifacts':observed,
           'upload_times':{row['filename']:row['upload_time_iso_8601'] for row in data['urls']},
           'index_json_sha256':hashlib.sha256((root/'pypi-raw.json').read_bytes()).hexdigest(),
           'github_release_url':'https://github.com/audunarn/ANYgeometry/releases/tag/v0.4.5',
           'publish_run':'https://github.com/audunarn/ANYgeometry/actions/runs/36842492356'}
(root/'pypi-receipt.json').write_bytes((json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode())
print(json.dumps(receipt))
