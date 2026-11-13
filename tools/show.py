import sys, ast
rows = []
for l in open(sys.argv[1]):
    l = l.strip()
    if l.startswith('{'): rows.append(ast.literal_eval(l))
    elif l: print(l)
rows.sort(key=lambda r: -r['ppm'])
for r in rows:
    extra = '  '.join('%s %.3f' % (k[:8], v) for k, v in r.items() if k not in ('name', 'ppm', 'gd'))
    print('%-10s ppm %.3f gd %+.3f  %s' % (r['name'], r['ppm'], r['gd'], extra))
