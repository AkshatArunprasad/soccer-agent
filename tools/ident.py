"""python ident.py A.py B.py OPP s0 s1 -> compares per-tick action streams of A and B (each vs OPP)."""
import sys, os, hashlib, importlib.util, time
SDK = '/tmp/claude-0/-home-user/0839d790-c0d4-5c6f-93b8-322acb9f80fc/scratchpad/proj/soccer-student-0.1.0/student-sdk'
sys.path.insert(0, SDK)
from concurrent.futures import ProcessPoolExecutor
from soccer.runner import play_match
from soccer import _engine
def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.MyTeam
def norm(a):
    out = []
    for pid in sorted(a):
        x = a[pid]
        if not isinstance(x, dict):
            x = {'movement': x.movement, 'kick_direction': x.kick_direction, 'kick_power': x.kick_power}
        mv = x.get('movement', (0, 0)); kd = x.get('kick_direction'); kp = x.get('kick_power', 0.0)
        out.append((pid, round(mv[0], 9), round(mv[1], 9), None if kd is None else (round(kd[0], 9), round(kd[1], 9)), round(kp, 9)))
    return out
def run(args):
    path, opp, seed, sw = args
    A = load(os.path.abspath(path), 'ta')
    h = hashlib.sha256(); tt = [0.0]
    class W(A):
        def act(self, obs):
            t = time.perf_counter(); a = super().act(obs); tt[0] += time.perf_counter() - t
            h.update(repr(norm(a)).encode())
            return a
    tb = opp if opp in _engine.baseline_names() else load(os.path.abspath(opp), 'tb')()
    ta = W()
    r = play_match(*((tb, ta) if sw else (ta, tb)), seed=seed, kickoff_team=0).result
    return seed, sw, h.hexdigest(), tuple(r['score']), tt[0]
if __name__ == '__main__':
    a, b, opp, s0, s1 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    jobs = [(p, opp, s, sw) for p in (a, b) for s in range(s0, s1 + 1) for sw in (0, 1)]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(run, jobs))
    n = len(res) // 2
    ra, rb = res[:n], res[n:]
    same = sum(1 for x, y in zip(ra, rb) if x[2] == y[2])
    print('identical action streams: %d / %d' % (same, n))
    for x, y in zip(ra, rb):
        if x[2] != y[2]: print('  differs: seed', x[0], 'sw', x[1], x[3], y[3])
    ta = sum(x[4] for x in ra); tb = sum(x[4] for x in rb)
    print('total act time A %.3fs  B %.3fs  ratio B/A %.3f' % (ta, tb, tb / ta))
