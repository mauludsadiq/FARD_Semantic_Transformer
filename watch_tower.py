import sys, json, subprocess, time, os

trace_path = sys.argv[1]
pid = int(sys.argv[2])

prev_ph = -1
seen = set()

def alive(pid):
    try:
        subprocess.check_output(["kill", "-0", str(pid)], stderr=subprocess.DEVNULL)
        return True
    except:
        return False

while True:
    with open(trace_path) as f:
        for i, line in enumerate(f):
            if i in seen: continue
            seen.add(i)
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                if d.get('t') == 'emit':
                    msg = d['v']['msg']
                    parts = dict(p.split('=',1) for p in msg.split(' ') if '=' in p)
                    ph  = int(parts.get('ph', 0))
                    blk = int(parts.get('blk', 0))
                    op  = parts.get('op', '?')
                    lyr = parts.get('layer', '?')
                    if ph != prev_ph:
                        if prev_ph >= 0: print()
                        print(f"phoneme {ph:3d}  ──────────────────────────────────", flush=True)
                        prev_ph = ph
                    bar = '█' * (blk + 1) + '░' * (11 - blk)
                    print(f"  [{bar}] blk={blk:2d}  {op:<20}  {lyr}", flush=True)
            except: pass

    if not alive(pid): break
    time.sleep(0.2)
