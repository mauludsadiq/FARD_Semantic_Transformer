#!/bin/bash
OUT=/tmp/tower_watch_$(date +%s)
echo "▶ FARD Semantic Tower — $OUT"
echo ""

./target/release/fardrun run \
  --program programs/semantic_transformer/tower.fard \
  --out $OUT 2>/dev/null &
PID=$!

while [ ! -f $OUT/trace.ndjson ]; do sleep 0.2; done

python3 watch_tower.py $OUT/trace.ndjson $PID

wait $PID
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 -c "
import json
with open('$OUT/result.json') as f:
    r = json.load(f)['result']
print(f\"  tower_root  {r['tower_root']}\")
print(f\"  total_ops   {r['total_ops']}\")
print(f\"  accepts     {r['accepts']}\")
print(f\"  model       {r['model']}\")
print(f\"  sequence    {r['sequence']}\")
"
