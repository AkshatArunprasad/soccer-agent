"""Classify goals: where the scoring kick was taken, kick type, prior sequence, time since kickoff."""
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
    out = []; st = {'kicks': [], 'ko': 0, 'ko_team': None}
    class W(A):
        def act(self, obs):
            bx, by = obs.ball.position
            for e in obs.events:
                if e['kind'] == 'kick':
                    st['kicks'].append((obs.tick, e['team'], e['power'], bx, by, e['player']))
                if e['kind'] == 'goal':
                    ks = [k for k in st['kicks'] if k[0] >= st['ko']]
                    last = ks[-1] if ks else None
                    scorer_team = e['team']
                    # last kick by scoring team
                    lk = next((k for k in reversed(ks) if k[1] == scorer_team), None)
                    out.append(dict(team=scorer_team, own=e.get('own_goal'), t=obs.tick - st['ko'], ko_team=st['ko_team'],
                                    lk=lk, nk=len(ks), opp_kicks=sum(1 for k in ks if k[1] != scorer_team)))
                    st['ko'] = obs.tick; st['ko_team'] = 1 - scorer_team
            if obs.tick == 0: st['ko_team'] = obs.ball.controlling_team
            return super().act(obs)
    ta = W(); tb = B()
    ea, eb = (tb, ta) if sw else (ta, tb)
    play_match(ea, eb, seed=seed, kickoff_team=0)
    return out
if __name__ == '__main__':
    a, b, s0, s1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    jobs = [(a, b, s, sw) for s in range(s0, s1 + 1) for sw in (0, 1)]
    allg = []
    with ProcessPoolExecutor(4) as ex:
        for o in ex.map(run, jobs): allg.extend(o)
    for team in (0, 1):
        gs = [g for g in allg if g['team'] == team]
        c = Counter()
        for g in gs:
            c['own'] += bool(g['own'])
            c['from_ko_by_scorer'] += g['ko_team'] == team
            c['within10s_of_ko'] += g['t'] <= 200
            c['ko_and_10s'] += (g['ko_team'] == team and g['t'] <= 200)
            if g['lk']:
                x = g['lk'][3]; d = math.hypot(50 - abs(x), g['lk'][4]) if True else 0
                gx = 50 if team == 0 else -50
                dist = math.hypot(gx - g['lk'][3], g['lk'][4])
                c['dist<12'] += dist < 12; c['dist12-18'] += 12 <= dist < 18; c['dist18-24'] += 18 <= dist < 24; c['dist>=24'] += dist >= 24
        print('US' if team == 0 else 'THEM', 'goals', len(gs), dict(c))
