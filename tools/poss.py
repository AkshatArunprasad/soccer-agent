import sys, os, math, importlib.util
SDK = '/tmp/claude-0/-home-user/0839d790-c0d4-5c6f-93b8-322acb9f80fc/scratchpad/proj/soccer-student-0.1.0/student-sdk'
sys.path.insert(0, SDK)
from concurrent.futures import ProcessPoolExecutor
from soccer.runner import play_match
from collections import Counter
def load(path, tag):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.MyTeam
def run(args):
    a, b, seed, sw = args
    A = load(os.path.abspath(a), 'ta'); B = load(os.path.abspath(b), 'tb')
    c = Counter(); st = {'owner': None, 'start': 0, 'lastkick': {0: None, 1: None}}
    def close(obs, reason):
        o = st['owner']
        if o is None: return
        dur = obs.tick - st['start']
        lk = st['lastkick'][o]
        kind = 'none' if lk is None else ('touch' if lk[0] < 0.5 else ('shot' if lk[1] else 'pass'))
        c[(('US' if o == 0 else 'THEM'), reason, kind)] += 1
        c[(('US' if o == 0 else 'THEM'), 'dur')] += dur
        c[(('US' if o == 0 else 'THEM'), 'n')] += 1
    class W(A):
        def act(self, obs):
            bx, by = obs.ball.position
            for e in obs.events:
                if e['kind'] == 'kick':
                    t = e['team']; d = e['direction']
                    gx = 50 if t == 0 else -50
                    toward = (d['x'] * (1 if t == 0 else -1)) > 0.3 and abs(gx - bx) < 30
                    st['lastkick'][t] = (e['power'], toward)
                if e['kind'] == 'goal':
                    close(obs, 'goal'); st['owner'] = None
                if e['kind'] == 'turnover':
                    close(obs, 'lost'); st['owner'] = e['to_team']; st['start'] = obs.tick; st['lastkick'] = {0: None, 1: None}
            if st['owner'] is None and obs.ball.controlling_team is not None and obs.ball.velocity == (0.0, 0.0):
                st['owner'] = obs.ball.controlling_team; st['start'] = obs.tick; st['lastkick'] = {0: None, 1: None}
            return super().act(obs)
    ta = W(); tb = B()
    ea, eb = (tb, ta) if sw else (ta, tb)
    play_match(ea, eb, seed=seed, kickoff_team=0)
    return c
if __name__ == '__main__':
    a, b, s0, s1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    jobs = [(a, b, s, sw) for s in range(s0, s1 + 1) for sw in (0, 1)]
    tot = Counter()
    with ProcessPoolExecutor(4) as ex:
        for c in ex.map(run, jobs): tot.update(c)
    n = len(jobs)
    for side in ('US', 'THEM'):
        k = tot[(side, 'n')]
        print(side, 'possessions/match', round(k / n, 2), 'mean ticks', round(tot[(side, 'dur')] / max(1, k), 1))
        for key, v in sorted(tot.items(), key=lambda kv: -kv[1]):
            if key[0] == side and len(key) == 3: print('   ', key[1:], round(v / n, 2))
