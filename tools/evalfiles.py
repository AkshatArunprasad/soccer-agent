"""python evalfiles.py team1.py,team2.py opp1,opp2 s0 s1  -> table of ppm per opponent"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tune
if __name__ == '__main__':
    teams = sys.argv[1].split(','); pool = sys.argv[2].split(','); s0, s1 = int(sys.argv[3]), int(sys.argv[4])
    poolp = [p if p in tune._engine.baseline_names() else os.path.join(tune.HERE, p) for p in pool]
    jobs = [(os.path.join(tune.HERE, t), o, s, sw) for t in teams for o in poolp for s in range(s0, s1 + 1) for sw in (0, 1)]
    t0 = time.time(); res = {}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        for a, b, s in ex.map(tune._run, jobs, chunksize=4):
            res.setdefault(a, {}).setdefault(b, []).append(s)
    for t in teams:
        a = os.path.join(tune.HERE, t); cells = []; tp = tn = 0
        for o in poolp:
            ss = res[a][o]; W = sum(1 for s in ss if s[0] > s[1]); D = sum(1 for s in ss if s[0] == s[1]); L = len(ss) - W - D
            cells.append('%s %.2f (%d-%d-%d)' % (os.path.basename(o)[:10], (3 * W + D) / len(ss), W, D, L)); tp += 3 * W + D; tn += len(ss)
        print('%-14s ppm %.3f | %s' % (t[:14], tp / tn, ' | '.join(cells)), flush=True)
    print('(%.0fs)' % (time.time() - t0))
