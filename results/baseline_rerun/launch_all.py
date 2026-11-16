#!/usr/bin/env python3
import subprocess, time, json, os, sys
from pathlib import Path

ROOT = Path('/workspace/quadremesh-fusion')
os.chdir(ROOT)
OUT = ROOT / 'results/baseline_rerun'
OUT.mkdir(exist_ok=True)
py = str(ROOT / '.venv/bin/python')

cases = [
  ('s07', [
    'tests/user_scans/s07.stl', '-o', str(OUT/'s07/s07.step'),
    '--report', str(OUT/'s07/s07.report.json'), '--device', 'cpu',
    '--time-budget', '3600', '--measure', 'bore_d=14.275',
    '--note', 'baseline_rerun s07 bore_d=14.275 (1a2e364/0c0a7ba)',
  ]),
  ('s09', [
    'tests/user_scans/s09.stl', '-o', str(OUT/'s09/s09.step'),
    '--report', str(OUT/'s09/s09.report.json'), '--device', 'cpu',
    '--time-budget', '3600', '--measure', 'bore_d=14.275',
    '--note', 'baseline_rerun s09 bore_d=14.275 (0c0a7ba/f602f5f)',
  ]),
  ('stg1', [
    'results/regression_loft_guard_2s/prepped/stg1_scale01.stl',
    '-o', str(OUT/'stg1/stg1.step'),
    '--report', str(OUT/'stg1/stg1.report.json'), '--device', 'cpu',
    '--time-budget', '3600', '--measure', 'bore_d=14.29',
    '--note', 'baseline_rerun stg1 scale0.1 bore_d=14.29 (66b2411)',
  ]),
  ('p01', [
    'tests/user_scans/p01.stl', '-o', str(OUT/'p01/p01.step'),
    '--report', str(OUT/'p01/p01.report.json'), '--device', 'cpu',
    '--time-budget', '900', '--gear-build', 'snapped',
    '--note', 'baseline_rerun p01 snapped (85694fb/1cbf88e)',
  ]),
  ('s05', [
    'results/regression_loft_guard_2s/prepped/s05.stl',
    '-o', str(OUT/'s05/s05.step'),
    '--report', str(OUT/'s05/s05.report.json'), '--device', 'cpu',
    '--time-budget', '600',
    '--note', 'baseline_rerun s05 prepped (historical prep_scan)',
  ]),
  ('s06', [
    'results/regression_loft_guard_2s/prepped/s06.stl',
    '-o', str(OUT/'s06/s06.step'),
    '--report', str(OUT/'s06/s06.report.json'), '--device', 'cpu',
    '--time-budget', '600',
    '--note', 'baseline_rerun s06 prepped (historical prep_scan)',
  ]),
]

# Smoke-import before fan-out
r = subprocess.run([py, '-c', 'import scipy, meshsolid; print("ok")'], capture_output=True, text=True)
print('smoke:', r.stdout, r.stderr, r.returncode, flush=True)
if r.returncode != 0:
    sys.exit(2)

procs = {}
for name, args in cases:
  (OUT/name).mkdir(exist_ok=True)
  log_path = OUT / f'{name}.log'
  log = open(log_path, 'w')
  log.write(f'=== START {name} {time.strftime("%Y-%m-%dT%H:%M:%S%z")} ===\n')
  log.flush()
  t0 = time.time()
  env = os.environ.copy()
  env['PYTHONUNBUFFERED'] = '1'
  p = subprocess.Popen([py, '-u', '-m', 'meshsolid', *args],
                       stdout=log, stderr=subprocess.STDOUT,
                       cwd=str(ROOT), env=env)
  procs[name] = {'p': p, 't0': t0, 'log': log, 'log_path': log_path}
  print(f'launched {name} pid={p.pid}', flush=True)
  time.sleep(1.5)  # stagger imports

state = {n: {'pid': v['p'].pid, 't0': v['t0']} for n, v in procs.items()}
(OUT/'launcher_state.json').write_text(json.dumps(state, indent=2))

summary = {}
# Poll so short cases finish first without blocking report of others
pending = set(procs)
while pending:
  done_now = []
  for name in list(pending):
    rc = procs[name]['p'].poll()
    if rc is not None:
      wall = time.time() - procs[name]['t0']
      procs[name]['log'].write(f'\n=== END {name} rc={rc} wall={wall:.1f}s ===\n')
      procs[name]['log'].close()
      summary[name] = {'rc': rc, 'wall_s': round(wall, 1)}
      (OUT/f'{name}.wall').write_text(f'{wall:.3f}\n')
      print(f'finished {name} rc={rc} wall={wall:.1f}s', flush=True)
      done_now.append(name)
  for name in done_now:
    pending.discard(name)
  (OUT/'summary_partial.json').write_text(json.dumps(summary, indent=2))
  if pending:
    time.sleep(5)

(OUT/'summary.json').write_text(json.dumps(summary, indent=2))
print('ALL DONE', summary, flush=True)
