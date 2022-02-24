import sys
import json

with open(sys.argv[1]) as f:
    j = json.load(f)

with open('tmp.json', 'w') as f:
    json.dump(j, f)
