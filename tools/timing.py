import sys, os, time, gc, importlib.util
SDK = '/tmp/claude-0/-home-user/0839d790-c0d4-5c6f-93b8-322acb9f80fc/scratchpad/proj/soccer-student-0.1.0/student-sdk'
sys.path.insert(0, SDK)
from soccer.runner import play_match
def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.MyTeam
A = load(os.path.abspath(sys.argv[1]), 'ta'); B = load(os.path.abspath(sys.argv[2]), 'tb')
times = []
class W(A):
    def act(self, obs):
        t = time.perf_counter()
        r = super().act(obs)
        times.append(((time.perf_counter() - t) * 1000, obs.tick))
        return r
for seed in range(int(sys.argv[3]), int(sys.argv[4])):
    for sw in (0, 1):
        ta = W(); tb = B()
        ea, eb = (tb, ta) if sw else (ta, tb)
        play_match(ea, eb, seed=seed, kickoff_team=0)
ts = sorted(t for t, _ in times)
n = len(ts)
print('n', n, 'mean %.4f' % (sum(ts) / n), 'p50 %.4f p99 %.3f p99.9 %.3f p99.99 %.3f max %.3f' % (ts[n // 2], ts[int(n * .99)], ts[int(n * .999)], ts[int(n * .9999)], ts[-1]))
#print('top', sorted(times, reverse=True)[:8])
