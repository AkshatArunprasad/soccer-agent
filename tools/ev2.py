"""python ev2.py TEAM.py OPP1,OPP2 s0 s1 [workers]  -- with goal-timeline stats."""
import sys, os, time, importlib.util
SDK = '/tmp/claude-0/-home-user/0839d790-c0d4-5c6f-93b8-322acb9f80fc/scratchpad/proj/soccer-student-0.1.0/student-sdk'
sys.path.insert(0, SDK)
from concurrent.futures import ProcessPoolExecutor
from soccer.runner import play_match
from soccer import _engine
_cache = {}
def load(path):
    if path not in _cache:
        spec = importlib.util.spec_from_file_location('m' + str(len(_cache)), path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        _cache[path] = m.MyTeam
    return _cache[path]
def run(args):
    a, b, seed, swapped = args
    A = load(os.path.abspath(a))
    goals = []
    class W(A):
        def act(self, obs):
            for e in obs.events:
                if e['kind'] == 'goal': goals.append((obs.tick, e['team']))
            return super().act(obs)
    ta = W()
    tb = b if b in _engine.baseline_names() else load(os.path.abspath(b))()
    ea, eb = (tb, ta) if swapped else (ta, tb)
    out = play_match(ea, eb, seed=seed, kickoff_team=0)
    r = out.result
    s = list(r['score']); st = r['stats']
    if swapped: s = s[::-1]; st = st[::-1]
    dm = st[0].get('decision_time_ms', {})
    return b, seed, swapped, s, dm.get('total_ms', 0), dm.get('count', 0), dm.get('max_ms', 0), st[0].get('policy_timeouts', 0), st[0].get('fouls', 0), goals
if __name__ == '__main__':
    team = sys.argv[1]; opps = sys.argv[2].split(',')
    s0, s1 = int(sys.argv[3]), int(sys.argv[4]); w = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    jobs = [(team, o, sd, sw) for o in opps for sd in range(s0, s1 + 1) for sw in (False, True)]
    t = time.time(); res = {}
    with ProcessPoolExecutor(w) as ex:
        for r in ex.map(run, jobs, chunksize=2):
            res.setdefault(r[0], []).append(r)
    tp = tn = 0
    for o in opps:
        rs = res[o]
        W = sum(1 for r in rs if r[3][0] > r[3][1]); L = sum(1 for r in rs if r[3][0] < r[3][1]); D = len(rs) - W - L
        gf = sum(r[3][0] for r in rs); ga = sum(r[3][1] for r in rs)
        ms = sum(r[4] for r in rs) / max(1, sum(r[5] for r in rs)); mx = max(r[6] for r in rs)
        fo = sum(r[8] for r in rs); to = sum(r[7] for r in rs)
        fs = [r for r in rs if r[9] and r[9][0][1] == 0]; fo_ = [r for r in rs if r[9] and r[9][0][1] == 1]
        fsw = sum(1 for r in fs if r[3][0] > r[3][1]); fsd = sum(1 for r in fs if r[3][0] == r[3][1])
        fow = sum(1 for r in fo_ if r[3][0] > r[3][1]); fod = sum(1 for r in fo_ if r[3][0] == r[3][1])
        nil = sum(1 for r in rs if not r[9])
        pts = 3 * W + D; tp += pts; tn += len(rs)
        print(f"{o:>14}: {W:4d}W {D:4d}D {L:4d}L GF {gf:4d} GA {ga:4d} ppm {pts/len(rs):.3f} | 1st-goal us {len(fs)} (W{fsw} D{fsd}) them {len(fo_)} (W{fow} D{fod}) 0-0 {nil} | ms {ms:.3f} max {mx:.1f} TO {to} F {fo}")
    print(f"TOTAL ppm {tp/tn:.3f} over {tn} ({time.time()-t:.0f}s)")
