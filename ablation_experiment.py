"""Experiment 2: Ablation Study (full / fdt_only / ldt_only / no_sync / full_sync)"""
import numpy as np
import matplotlib.pyplot as plt
from simulation_core import run_simulation, DDTScheduler

NUM_RUNS = 5

def main():
    rates = [20, 50, 100, 150, 200, 250, 300]

    modes = [
        ('full',      'red',    'o',  '-'),
        ('fdt_only',  'blue',   's',  '--'),
        ('ldt_only',  'green',  'v',  '--'),
        ('no_sync',   'gray',   'x',  ':'),
        ('full_sync', 'purple', 'D',  '-'),
    ]

    results = {m: {'lat':[], 'suc':[], 'ov':[]} for m,_,_,_ in modes}

    print("Running Ablation Study...")
    for rate in rates:
        print(f"  Rate = {rate}")
        for mode, _, _, _ in modes:
            lats, sucs, ovs = [], [], []
            for _ in range(NUM_RUNS):
                l, s, o = run_simulation(DDTScheduler, rate, mode=mode)
                lats.append(l); sucs.append(s); ovs.append(o)
            results[mode]['lat'].append(np.mean(lats))
            results[mode]['suc'].append(np.mean(sucs))
            results[mode]['ov'].append(np.mean(ovs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for mode, color, marker, ls in modes:
        axes[0].plot(rates, results[mode]['lat'], label=mode, color=color, marker=marker, ls=ls, markersize=5)
        axes[1].plot(rates, results[mode]['suc'], label=mode, color=color, marker=marker, ls=ls, markersize=5)
        axes[2].plot(rates, results[mode]['ov'],  label=mode, color=color, marker=marker, ls=ls, markersize=5)

    axes[0].set_title('(a) Average Latency')
    axes[0].set_xlabel('Task Arrival Rate (tasks/min/sat)')
    axes[0].set_ylabel('Avg Latency (s)')
    axes[0].legend(fontsize=7); axes[0].grid(True)

    axes[1].set_title('(b) Success Rate')
    axes[1].set_xlabel('Task Arrival Rate (tasks/min/sat)')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].legend(fontsize=7); axes[1].grid(True)

    axes[2].set_title('(c) Overhead')
    axes[2].set_xlabel('Task Arrival Rate (tasks/min/sat)')
    axes[2].set_ylabel('Overhead (Bytes/s)')
    axes[2].legend(fontsize=7); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('ablation_study.pdf', dpi=300)
    plt.savefig('ablation_study.png', dpi=300)
    print("Saved ablation_study.pdf / .png")

if __name__ == '__main__':
    main()
