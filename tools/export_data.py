"""Export the numbers behind the paper figures to assets/data.js for the web charts.

Reads the CSV files of the paper repository (plots/data) and the constants of plot_real_exec.py.
Usage: python3 tools/export_data.py [path/to/ODE_Policy_paper/plots/data]
"""
import csv, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'ODE_Policy_paper', 'plots', 'data')
SU = ('spatial', 'object', 'goal', 'long')


def read(name):
    with open(os.path.join(DATA, name), encoding='utf-8') as f:
        return list(csv.DictReader(l for l in f if not l.startswith('#')))


def num(v):
    return round(float(v), 4)


def pstd(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


out = {}

# LIBERO learning curves, mean and std over three seeds
d = defaultdict(list)
for r in read('e1_mean.csv'):
    d[r['method']].append([int(r['epoch']), num(r['mean']), num(r['std'])])
out['libero'] = {m: sorted(v) for m, v in d.items()}

# data and parameter efficiency, ODE vs. FM-DiT
for key, name, col in (('demos', 'e3_mean.csv', 'demos'), ('size', 'e4_mean.csv', 'size')):
    d = defaultdict(list)
    for r in read(name):
        eps = [k[2:] for k in r if k.startswith('ep')]
        d[r['method']].append({'x': r[col], **{e: [num(r['ep' + e]), num(r['sd' + e])] for e in eps}})
    out[key] = d

# VLA (pi0.5) and WAM (ImageWAM), one seed, band = std over the four suites
d = defaultdict(list)
for r in read('vla_pi05.csv'):
    d[r['bench'] + ':' + r['method']].append([num(r['epoch']), num(r['value']), num(r['suite_sd'])])
out['vla'] = {k: sorted(v) for k, v in d.items()}
w = defaultdict(dict)
for r in read('wam_imagewam.csv'):
    w[(r['bench'], r['method'], int(r['epoch']))][r['suite']] = float(r['mean'])
d = defaultdict(list)
for (b, m, e), s in w.items():
    d[b + ':' + m].append([e, round(s['all'], 2), round(pstd([s[k] for k in SU]), 2)])
out['wam'] = {k: sorted(v) for k, v in d.items()}

# RL, SuperMimic on the 40 LAFAN1 clips with a G1 humanoid, over wall-clock hours (three random seeds)
tr = read('rl_supermimic_training_return_3way.csv')
out['rl_train'] = {m: [[num(r['hours']), num(r[m + '_mean']), num(r[m + '_std'])] for r in tr[::6]] for m in ('ode', 'mlp', 'ppo')}
out['rl_eval'] = {m: [[num(r['hours_mean']), num(r['full_return_mean']), num(r['full_return_std']), num(r['action_acc_mean']) * 1000,
                       num(r['action_acc_std']) * 1000] for r in read(f'rl_supermimic_eval_hours_{m}.csv')] for m in ('ode', 'mlp', 'ppo')}

# inference time vs. success on LIBERO, smoothness over epochs
out['time'] = {r['method']: [num(r['ms_call']), num(r['ms_call_std']), num(r['sr']), num(r['sr_sd'])] for r in read('e_time.csv')}
d = defaultdict(list)
for r in read('e2_exec.csv'):
    d[r['method']].append([int(r['epoch']), num(r['jerk']), num(r['jerk_std'])])
out['jerk'] = d
out['vla_exec'] = {r['method']: {k: num(r[k]) for k in ('model', 'head', 'cmd_in', 'cmd_bd', 'action_params_M')} for r in read('vla_exec.csv')}

# real robot, one task per robot (constants of plot_real_exec.py): close box (Franka), microwave (Galaxea R1 Lite),
# can (Sharpa hand); [mean, std], the hand reports the inference time without a std
out['real'] = {
    'single': {'order': ['DDPM-UNet', 'DDIM-UNet', 'FM-DiT', 'ODE'], 'gpu': 'RTX 5080',
               'time': {'DDPM-UNet': [408.0, 5.3], 'DDIM-UNet': [83.4, 2.1], 'FM-DiT': [64.5, 1.3], 'ODE': [15.1, 0.5]},
               'acc_bd': {'DDPM-UNet': [25.2, 0.84], 'DDIM-UNet': [26.0, 1.2], 'FM-DiT': [20.9, 1.1], 'ODE': [3.58, 0.30]},
               'acc_in': {'DDPM-UNet': [7.44, 0.36], 'DDIM-UNet': [10.5, 0.71], 'FM-DiT': [8.86, 0.63], 'ODE': [0.94, 0.05]},
               'acc_demo': [2.60, 2.62], 'acc_max': 30,
               'jerk': {'DDPM-UNet': [539, 43], 'DDIM-UNet': [270, 7], 'FM-DiT': [211, 10], 'ODE': [45, 2]}, 'jerk_demo': 44.7, 'jerk_max': 600},
    'dual': {'order': ['DDPM-UNet', 'DDIM-UNet', 'FM-DiT', 'ODE'], 'gpu': 'RTX 5090',
             'time': {'DDPM-UNet': [223.1, 2.4], 'DDIM-UNet': [38.0, 1.0], 'FM-DiT': [31.0, 0.6], 'ODE': [8.4, 0.2]},
             'acc_bd': {'DDPM-UNet': [6.416, 0.239], 'DDIM-UNet': [5.822, 0.873], 'FM-DiT': [2.396, 0.314], 'ODE': [0.459, 0.046]},
             'acc_in': {'DDPM-UNet': [2.303, 0.505], 'DDIM-UNet': [2.985, 0.658], 'FM-DiT': [0.594, 0.058], 'ODE': [0.158, 0.025]},
             'acc_demo': [0.443, 0.446], 'acc_max': 7.5,
             'jerk': {'DDPM-UNet': [19.9, 1.6], 'DDIM-UNet': [23.2, 1.7], 'FM-DiT': [12.1, 1.0], 'ODE': [7.9, 0.9]}, 'jerk_demo': 8.5, 'jerk_max': 26},
    'hand': {'order': ['DDIM-UNet', 'FM-DiT', 'ODE'], 'gpu': 'RTX 4090',
             'time': {'DDIM-UNet': [53.1, 0], 'FM-DiT': [36.3, 0], 'ODE': [10.3, 0]},
             'acc_bd': {'DDIM-UNet': [5.08, 1.96], 'FM-DiT': [4.46, 0.42], 'ODE': [0.49, 0.13]},
             'acc_in': {'DDIM-UNet': [2.27, 0.90], 'FM-DiT': [1.50, 0.14], 'ODE': [0.40, 0.04]},
             'acc_demo': [0.72, 0.65], 'acc_max': 7.5,
             'jerk': {'DDIM-UNet': [13.3, 0.8], 'FM-DiT': [15.6, 1.3], 'ODE': [6.0, 0.6]}, 'jerk_demo': 7.6, 'jerk_max': 18},
}
# success rates (%) over 20 trials of all real-robot tasks (plot_real_franka.py, plot_real_dual_rows.py,
# plot_real_hand_rows.py); the two-flower task counts trials with at least one flower inserted
out['tasks'] = {
    'box': {'ODE': 95, 'A2A-DiT': 75, 'DDPM-UNet': 60, 'FM-DiT': 50, 'DDIM-UNet': 40},
    'key': {'ODE': 90, 'A2A-DiT': 75, 'DDIM-UNet': 35, 'FM-DiT': 35, 'DDPM-UNet': 30},
    'sock': {'ODE': 75, 'FM-DiT': 40, 'A2A-DiT': 30, 'DDPM-UNet': 20, 'DDIM-UNet': 15},
    'insert': {'ODE': 70, 'FM-DiT': 20, 'DDPM-UNet': 10, 'DDIM-UNet': 5, 'A2A-DiT': 0},
    'poppy': {'ODE': 30, 'FM-DiT': 20, 'DDPM-UNet': 0, 'DDIM-UNet': 0},
    'dahlia': {'ODE': 60, 'FM-DiT': 40, 'DDIM-UNet': 5, 'DDPM-UNet': 0},
    'microwave': {'ODE': 75, 'FM-DiT': 20, 'DDPM-UNet': 0, 'DDIM-UNet': 0},
    'flowers': {'ODE': 55, 'FM-DiT': 20, 'DDPM-UNet': 0, 'DDIM-UNet': 0},
    'drawer': {'ODE': 60, 'FM-DiT': 0},
    'can': {'ODE': 90, 'FM-DiT': 70, 'DDIM-UNet': 40},
    'cup': {'ODE': 85, 'FM-DiT': 0, 'DDIM-UNet': 0},
    'pills': {'ODE': 85, 'FM-DiT': 50, 'DDIM-UNet': 0},
    'cube': {'ODE': 65, 'FM-DiT': 0, 'DDIM-UNet': 0},
}

# LIBERO-plus zero-shot
d = defaultdict(list)
for r in read('e6_plus.csv'):
    d[r['method']].append([int(r['epoch']), num(r['overall']), num(r['overall_sd'])])
out['plus'] = d

# 2D navigation
d = defaultdict(list)
for r in read('e_nav2d.csv'):
    d[r['method']].append([int(r['epoch']), num(r['sr']), num(r['sr_std']), num(r['gamma']), num(r['gamma_std'])])
out['nav'] = d

js = '// generated by tools/export_data.py from the paper data, do not edit\nwindow.ODE_DATA = ' + json.dumps(out, separators=(',', ':')) + ';\n'
open(os.path.join(HERE, '..', 'assets', 'data.js'), 'w', encoding='utf-8').write(js)
print('assets/data.js', len(js))
