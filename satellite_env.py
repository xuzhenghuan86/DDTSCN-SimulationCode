"""Gymnasium environment for PPO training on the satellite network."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from simulation_core import (
    NUM_SATELLITES, SAT_PROCESSING_RATE, SATS_PER_PLANE,
    Satellite, Task, calculate_delay_s, SIM_DURATION_S, TIME_STEP_S
)

class SatelliteEnv(gym.Env):
    def __init__(self, task_arrival_rate=200):
        super().__init__()
        self.action_space = spaces.Discrete(NUM_SATELLITES)
        # 25 loads + 4 task features = 29
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(NUM_SATELLITES + 4,), dtype=np.float32)
        self.task_arrival_rate = task_arrival_rate
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.satellites = [
            Satellite(i, i // SATS_PER_PLANE, i % SATS_PER_PLANE,
                      SAT_PROCESSING_RATE)
            for i in range(NUM_SATELLITES)]
        self.current_time_s = 0
        self.task_id = 0
        self.rate_per_step = self.task_arrival_rate / 60.0
        self.pending = []
        self._advance_until_task()
        return self._obs(), {}

    def _advance_until_task(self):
        while not self.pending and self.current_time_s < SIM_DURATION_S:
            for sat in self.satellites:
                sat.update_position(self.current_time_s)
                if sat.is_processing and sat.current_task:
                    sat.current_task.cpu_cycles -= sat.processing_rate * TIME_STEP_S
                    if sat.current_task.cpu_cycles <= 0:
                        sat.is_processing = False
                        sat.current_task = None
                if not sat.is_processing and sat.task_queue:
                    nxt = sat.task_queue[0]
                    if nxt.arrival_time_s <= self.current_time_s:
                        sat.current_task = sat.task_queue.pop(0)
                        sat.is_processing = True
            n = np.random.poisson(self.rate_per_step * TIME_STEP_S)
            for _ in range(n):
                self.task_id += 1
                src = random.randint(0, NUM_SATELLITES - 1)
                self.pending.append(Task(self.task_id, self.current_time_s, src))
            if not self.pending:
                self.current_time_s += TIME_STEP_S

    def _obs(self):
        max_load = 500e6
        loads = [s.get_queue_load() / max_load for s in self.satellites]
        if self.pending:
            t = self.pending[0]
            feats = [t.category, t.size_mb / 50.0,
                     t.cpu_cycles / 500e6, t.e2e_deadline_s / 55.0]
        else:
            feats = [0.0, 0.0, 0.0, 0.0]
        return np.clip(np.array(loads + feats, dtype=np.float32), 0, 1.0)

    def step(self, action):
        if not self.pending:
            return self._obs(), 0, True, False, {}
        task = self.pending.pop(0)
        sender = self.satellites[task.sat_id]
        target = self.satellites[action]

        offload_delay = calculate_delay_s(sender, target, task.size_mb)
        queue_delay = target.get_queue_wait_time()
        est_delay = offload_delay + queue_delay

        # Heterogeneous reward
        if task.category == 0:
            reward = -est_delay
            if est_delay > task.e2e_deadline_s:
                reward -= 50 + ((est_delay / task.e2e_deadline_s) ** 2) * 10
        else:
            reward = (10 - est_delay * 0.2) if est_delay <= task.e2e_deadline_s \
                     else (-est_delay - 20)

        task.arrival_time_s += offload_delay
        target.add_task(task)

        if not self.pending:
            self.current_time_s += TIME_STEP_S
            self._advance_until_task()

        done = self.current_time_s >= SIM_DURATION_S
        return self._obs(), reward, done, False, {}
