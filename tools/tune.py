"""Evaluate parameter variants of a base team against an opponent pool.
Usage (as module): run_variants(base, [ (name, {CONST: value}), ...], pool, seeds)"""
import sys, os, re, time, json, importlib.util
SDK = '/tmp/claude-0/-home-user/0839d790-c0d4-5c6f-93b8-322acb9f80fc/scratchpad/proj/soccer-student-0.1.0/student-sdk'
sys.path.insert(0, SDK)
from concurrent.futures import ProcessPoolExecutor
from soccer.runner import play_match
from soccer import _engine
HERE = os.path.dirname(os.path.abspath(__file__))
_cache = {}
def load(path):
    if path not in _cache:
        spec = importlib.util.spec_from_file_location('m%d' % len(_cache), path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _cache[path] = m.MyTeam
    return _cache[path]
def make_variant(base, name, params):
    src = open(os.path.join(HERE, base)).read()
    for k, v in params.items():
        pat = re.compile(r'^%s = .*$' % re.escape(k), re.M)
        assert pat.search(src), k
        src = pat.sub('%s = %r' % (k, v), src, count=1)
    path = os.path.join(HERE, 'var', name + '.py')
    open(path, 'w').write(src)
    return path
def _run(args):
    a, b, seed, sw = args
    ta = load(a)()
    tb = b if b in _engine.baseline_names() else load(b)()
    ea, eb = (tb, ta) if sw else (ta, tb)
    r = play_match(ea, eb, seed=seed, kickoff_team=0).result
    s = list(r['score'])
    if sw: s = s[::-1]
    return a, b, s
def run_variants(base, variants, pool, seeds, workers=4, quiet=False):
    paths = [(n, make_variant(base, n, p)) for n, p in variants]
    pool = [p if p in _engine.baseline_names() else os.path.join(HERE, p) for p in pool]
    jobs = [(path, o, s, sw) for _, path in paths for o in pool for s in seeds for sw in (0, 1)]
    t = time.time()
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for a, b, s in ex.map(_run, jobs, chunksize=4):
            res.setdefault(a, {}).setdefault(b, []).append(s)
    out = []
    for n, path in paths:
        row = {'name': n}
        tp = tn = tgd = 0
        for o in pool:
            ss = res[path][o]
            W = sum(1 for s in ss if s[0] > s[1]); D = sum(1 for s in ss if s[0] == s[1])
            pts = 3 * W + D; gd = sum(s[0] - s[1] for s in ss)
            row[os.path.basename(o)] = round(pts / len(ss), 3)
            tp += pts; tn += len(ss); tgd += gd
        row['ppm'] = round(tp / tn, 3); row['gd'] = round(tgd / tn, 3)
        out.append(row)
        if not quiet: print(row, flush=True)
    if not quiet: print('(%.0fs)' % (time.time() - t), flush=True)
    return out
if __name__ == '__main__':
    base = sys.argv[1]; spec = json.load(open(sys.argv[2][1:])) if sys.argv[2].startswith('@') else json.loads(sys.argv[2]); pool = sys.argv[3].split(','); s0, s1 = int(sys.argv[4]), int(sys.argv[5])
    run_variants(base, [(n, p) for n, p in spec], pool, range(s0, s1 + 1))
