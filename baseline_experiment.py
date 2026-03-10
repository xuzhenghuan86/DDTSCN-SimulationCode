"""Experiment 1: 7-Baseline Comparison (Latency / Success / Overhead vs Rate)"""
import numpy as np
import matplotlib.pyplot as plt
from simulation_core import (
    run_simulation, DDTScheduler, OnlineDRLScheduler, OfflineDRLScheduler,
    OptimalScheduler, GreedyScheduler, LocalScheduler, RandomScheduler
)

NUM_RUNS = 5

def main():
    rates = [20, 50, 100, 150, 200, 250, 300]

    schedulers = [
        ('Optimal',    OptimalScheduler,    {},              'black',  '+', '-'),
        ('DDT-SCN',    DDTScheduler,        {'mode':'full'}, 'red',    'o', '-'),
        ('Online-DRL', OnlineDRLScheduler,  {},              'blue',   's', '-'),
        ('Offline-DRL',OfflineDRLScheduler, {},              'green',  '^', '-'),
        ('Greedy',     GreedyScheduler,     {},              'purple', 'v', '-'),
        ('Random',     RandomScheduler,     {},              'orange', 'D', '-'),
        ('Local',      LocalScheduler,      {},              'gray',   'x', '-'),
    ]

    results = {n: {'lat':[], 'suc':[], 'ov':[]} for n,_,_,_,_,_ in schedulers}

    print("Running Baseline Comparison Experiment...")
    for rate in rates:
        print(f"  Rate = {rate}")
        for name, cls, kw, _, _, _ in schedulers:
            lats, sucs, ovs = [], [], []
            for _ in range(NUM_RUNS):
                l, s, o = run_simulation(cls, rate, **kw)
                lats.append(l); sucs.append(s); ovs.append(o)
            results[name]['lat'].append(np.mean(lats))
            results[name]['suc'].append(np.mean(sucs))
            results[name]['ov'].append(np.mean(ovs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, _, _, color, marker, ls in schedulers:
        axes[0].plot(rates, results[name]['lat'], label=name, color=color, marker=marker, ls=ls, markersize=5)
        axes[1].plot(rates, results[name]['suc'], label=name, color=color, marker=marker, ls=ls, markersize=5)
        axes[2].plot(rates, results[name]['ov'],  label=name, color=color, marker=marker, ls=ls, markersize=5)

    axes[0].set_title('Latency Comparison')
    axes[0].set_xlabel('Task Arrival Rate (tasks/min/sat)')
    axes[0].set_ylabel('Avg Latency (s)')
    axes[0].legend(fontsize=6); axes[0].grid(True)

    axes[1].set_title('Success Rate Comparison')
    axes[1].set_xlabel('Task Arrival Rate (tasks/min/sat)')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].legend(fontsize=6); axes[1].grid(True)

    axes[2].set_title('Comm. Overhead')
    axes[2].set_xlabel('Task Arrival Rate (tasks/min/sat)')
    axes[2].set_ylabel('Overhead (Bytes/step)')
    axes[2].legend(fontsize=6); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('baseline_comparison.pdf', dpi=300)
    plt.savefig('baseline_comparison.png', dpi=300)
    print("Saved baseline_comparison.pdf / .png")

if __name__ == '__main__':
    main()
