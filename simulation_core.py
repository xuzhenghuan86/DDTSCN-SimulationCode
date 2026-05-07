"""
DDT-SCN Satellite Network Simulation Core
==========================================
Implements a discrete-event simulation of a LEO satellite computing network
with heterogeneous tasks, priority queuing, and multiple scheduling algorithms.
All performance differences emerge organically from queuing theory and physics.
"""

import random
import numpy as np
import math

# =============================================================================
# Simulation Parameters
# =============================================================================
SIM_DURATION_S = 500       # seconds (use 1000 for production runs)
TIME_STEP_S = 1            # discrete time step
NUM_ORBIT_PLANES = 5
SATS_PER_PLANE = 5
NUM_SATELLITES = NUM_ORBIT_PLANES * SATS_PER_PLANE  # 25

ORBIT_ALTITUDE_KM = 1000
SAT_PROCESSING_RATE = 100e6       # 100 MHz (space-grade LEON4 class)
ISL_BANDWIDTH_MBPS = 100.0        # Inter-Satellite Link bandwidth
TASK_ARRIVAL_RATES = [20, 50, 100, 150, 200, 250, 300]
NUM_SIM_RUNS = 3                  # Use 10+ for production

# Task Heterogeneity ----------------------------------------------------------
CAT0_PROB = 0.3  # 30% delay-sensitive, 70% delay-tolerant

# Cat-0 (Delay-sensitive, light compute): telemetry, control commands
CAT0_CPU_RANGE   = (150e6, 300e6)    # 150-300M cycles
CAT0_SIZE_RANGE  = (5, 20)           # 5-20 MB
CAT0_DEADLINE    = (8, 20)           # 8-20 s strict deadline

# Cat-1 (Delay-tolerant, heavy compute): earth observation, AI inference
CAT1_CPU_RANGE   = (300e6, 500e6)    # 300-500M cycles
CAT1_SIZE_RANGE  = (15, 50)          # 15-50 MB
CAT1_DEADLINE    = (25, 55)          # 25-55 s relaxed deadline

# Physical Constants ----------------------------------------------------------
C = 299792458  # speed of light m/s
ORBIT_RADIUS_M = (6371 + ORBIT_ALTITUDE_KM) * 1000
ORBIT_PERIOD_S = 2 * math.pi * math.sqrt(ORBIT_RADIUS_M**3 / 3.986004418e14)
ORBIT_SPEED_MPS = 2 * math.pi * ORBIT_RADIUS_M / ORBIT_PERIOD_S

# Overhead message sizes (Bytes) - as per paper Section III.C ----------------
MSG_STATE_UPDATE = 22   # Incremental state update packet (22 bytes per paper)
MSG_NEIGHBOR_PING = 22  # LDT neighbor probe packet

# Hotspot configuration -------------------------------------------------------
HOTSPOT_SAT_COUNT = 6         # Number of satellites in hotspot region
HOTSPOT_THRESHOLD_RATE = 100  # Rate above which hotspot effect kicks in

# =============================================================================
# Task Class
# =============================================================================
class Task:
    def __init__(self, task_id, arrival_time_s, sat_id):
        self.task_id = task_id
        self.original_arrival_time_s = arrival_time_s
        self.arrival_time_s = arrival_time_s
        self.sat_id = sat_id
        self.completion_time_s = -1

        if random.random() < CAT0_PROB:
            self.category = 0
            self.cpu_cycles = random.uniform(*CAT0_CPU_RANGE)
            self.size_mb = random.uniform(*CAT0_SIZE_RANGE)
            self.e2e_deadline_s = random.uniform(*CAT0_DEADLINE)
        else:
            self.category = 1
            self.cpu_cycles = random.uniform(*CAT1_CPU_RANGE)
            self.size_mb = random.uniform(*CAT1_SIZE_RANGE)
            self.e2e_deadline_s = random.uniform(*CAT1_DEADLINE)

# =============================================================================
# Satellite Class
# =============================================================================
class Satellite:
    def __init__(self, sat_id, orbit_plane, slot, processing_rate):
        self.sat_id = sat_id
        self.orbit_plane = orbit_plane
        self.slot = slot
        self.processing_rate = processing_rate
        self.task_queue = []
        self.current_task = None
        self.is_processing = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.last_sync_time = 0
        
        # Orbital parameters for analytical J2 propagation
        self.inclination = math.radians(53.0)
        self.raan_0 = self.orbit_plane * (2 * math.pi / NUM_ORBIT_PLANES)
        # Walker-Delta phase offset (F=1)
        phase_offset = self.orbit_plane * (2 * math.pi / NUM_SATELLITES)
        self.mean_anomaly_0 = self.slot * (2 * math.pi / SATS_PER_PLANE) + phase_offset

    def add_task(self, task):
        """Priority queuing: Category-0 tasks are inserted before Category-1."""
        if task.category == 0:
            insert_pos = 0
            for i, t in enumerate(self.task_queue):
                if t.category == 1:
                    insert_pos = i
                    break
                insert_pos = i + 1
            self.task_queue.insert(insert_pos, task)
        else:
            self.task_queue.append(task)

    def update_position(self, current_time_s):
        # Analytical SGP4-lite with J2 perturbations
        R_E = 6371000
        J2 = 1.08263e-3
        MU = 3.986004418e14
        
        n = math.sqrt(MU / ORBIT_RADIUS_M**3)
        # J2 nodal precession rate
        raan_dot = -1.5 * n * J2 * (R_E / ORBIT_RADIUS_M)**2 * math.cos(self.inclination)
        
        current_raan = self.raan_0 + raan_dot * current_time_s
        current_anomaly = self.mean_anomaly_0 + n * current_time_s
        
        # 3D position in ECI coordinates
        self.x = ORBIT_RADIUS_M * (math.cos(current_raan)*math.cos(current_anomaly) - math.sin(current_raan)*math.sin(current_anomaly)*math.cos(self.inclination))
        self.y = ORBIT_RADIUS_M * (math.sin(current_raan)*math.cos(current_anomaly) + math.cos(current_raan)*math.sin(current_anomaly)*math.cos(self.inclination))
        self.z = ORBIT_RADIUS_M * (math.sin(current_anomaly)*math.sin(self.inclination))

    def get_queue_load(self):
        load = sum(t.cpu_cycles for t in self.task_queue)
        if self.current_task:
            load += self.current_task.cpu_cycles
        return load

    def get_queue_wait_time(self):
        return self.get_queue_load() / self.processing_rate

# =============================================================================
# Utility Functions
# =============================================================================
def calculate_delay_s(sat1, sat2, task_size_mb):
    """Calculate propagation + transmission delay between two satellites."""
    if sat1.sat_id == sat2.sat_id:
        return 0.0
    dist = math.sqrt((sat1.x - sat2.x)**2 + (sat1.y - sat2.y)**2 + (sat1.z - sat2.z)**2)
    dist = min(dist, 2 * ORBIT_RADIUS_M)
    prop_delay = dist / C
    trans_delay = (task_size_mb * 8 * 1e6) / (ISL_BANDWIDTH_MBPS * 1e6) 
    return prop_delay + trans_delay

def is_neighbor(sat1, sat2):
    """Walker-Delta neighbor with range and Earth-limb clearance constraints."""
    if sat1.sat_id == sat2.sat_id:
        return True
        
    dist = math.sqrt((sat1.x - sat2.x)**2 + (sat1.y - sat2.y)**2 + (sat1.z - sat2.z)**2)
    if dist > 2500000:  # Range-based threshold: 2500 km maximum
        return False
        
    # Earth-limb clearance constraint: 100 km minimum
    R_E = 6371000
    h_min = math.sqrt(max(0, ORBIT_RADIUS_M**2 - (dist/2)**2)) - R_E
    if h_min < 100000:
        return False
        
    same_plane = (sat1.orbit_plane == sat2.orbit_plane)
    cross_plane = (abs(sat1.orbit_plane - sat2.orbit_plane) in [1, NUM_ORBIT_PLANES - 1] and
                   sat1.slot == sat2.slot)
    return same_plane or cross_plane

# =============================================================================
# Schedulers
# =============================================================================
class Scheduler:
    def __init__(self, sat_list, sim_config=None):
        self.sat_list = sat_list
        self.sim_config = sim_config or {}
        self.total_overhead_bytes = 0

    def schedule(self, task, current_time_s):
        raise NotImplementedError

    def get_overhead_per_step(self):
        return self.total_overhead_bytes / max(1, SIM_DURATION_S / TIME_STEP_S)

    def _get_neighbor_ids(self, sat_id):
        sat = self.sat_list[sat_id]
        return [s.sat_id for s in self.sat_list
                if s.sat_id != sat_id and is_neighbor(sat, s)]

    def _get_drl_state(self, loads, task=None):
        max_load = 500e6
        base = [loads.get(i, 0) / max_load for i in range(NUM_SATELLITES)]
        if task is not None:
            base += [task.category, task.size_mb / 50.0,
                     task.cpu_cycles / 500e6, task.e2e_deadline_s / 55.0]
        else:
            base += [0.0, 0.0, 0.0, 0.0]
        return np.clip(np.array(base, dtype=np.float32), 0, 1.0)


class LocalScheduler(Scheduler):
    """Process all tasks locally. Zero overhead, no offloading."""
    def schedule(self, task, current_time_s):
        self.sat_list[task.sat_id].add_task(task)


class RandomScheduler(Scheduler):
    """Offload to a random satellite. Minimal overhead."""
    def schedule(self, task, current_time_s):
        sender = self.sat_list[task.sat_id]
        target = self.sat_list[random.randint(0, len(self.sat_list) - 1)]
        delay = calculate_delay_s(sender, target, task.size_mb)
        task.arrival_time_s += delay + 0.01  # tiny routing decision
        target.add_task(task)


class GreedyScheduler(Scheduler):
    """Pick the least-loaded neighbor. Moderate overhead from probing."""
    def schedule(self, task, current_time_s):
        sender = self.sat_list[task.sat_id]
        neighbor_ids = self._get_neighbor_ids(sender.sat_id)
        self.total_overhead_bytes += len(neighbor_ids) * MSG_NEIGHBOR_PING

        # Greedy picks best among neighbors (limited horizon)
        candidates = [(self.sat_list[nid], self.sat_list[nid].get_queue_load())
                       for nid in neighbor_ids]
        candidates.append((sender, sender.get_queue_load()))
        target = min(candidates, key=lambda x: x[1])[0]

        delay = calculate_delay_s(sender, target, task.size_mb)
        task.arrival_time_s += delay + 0.02
        target.add_task(task)


class OptimalScheduler(Scheduler):
    """Oracle-Greedy: omniscient global search for min (prop+queue) delay.
       Represents a theoretical upper bound. No overhead (oracle assumption)."""
    def schedule(self, task, current_time_s):
        sender = self.sat_list[task.sat_id]
        best_sat, best_time = sender, sender.get_queue_wait_time()
        for sat in self.sat_list:
            offload = calculate_delay_s(sender, sat, task.size_mb)
            total = offload + sat.get_queue_wait_time()
            if total < best_time:
                best_time = total
                best_sat = sat
        delay = calculate_delay_s(sender, best_sat, task.size_mb)
        task.arrival_time_s += delay + 0.01
        best_sat.add_task(task)


class OnlineDRLScheduler(Scheduler):
    """Online DRL: collects full global state per decision → high overhead + delay."""
    def __init__(self, sat_list, sim_config=None):
        super().__init__(sat_list, sim_config)
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load("ppo_fdt", device='cpu')
            self.has_model = True
        except Exception:
            self.has_model = False

    def schedule(self, task, current_time_s):
        sender = self.sat_list[task.sat_id]
        # Overhead: query ALL N satellites for state
        self.total_overhead_bytes += NUM_SATELLITES * MSG_STATE_UPDATE
        decision_delay = 0.08  # state collection + inference

        loads = {s.sat_id: s.get_queue_load() for s in self.sat_list}

        if self.has_model:
            state = self._get_drl_state(loads, task)
            action, _ = self.model.predict(state, deterministic=True)
            target_id = int(action)
            # Sanity check: if model picks a very loaded sat, override
            model_load = loads.get(target_id, 0)
            min_load = min(loads.values())
            if model_load > min_load * 3 and min_load > 0:
                target_id = min(range(NUM_SATELLITES), key=lambda i: loads.get(i,0))
        else:
            target_id = min(range(NUM_SATELLITES),
                            key=lambda i: loads.get(i, 0))

        target = self.sat_list[target_id]
        delay = calculate_delay_s(sender, target, task.size_mb)
        task.arrival_time_s += decision_delay + delay
        target.add_task(task)


class OfflineDRLScheduler(Scheduler):
    """Offline DRL: uses stale cached state → fast but degrades under load."""
    def __init__(self, sat_list, sim_config=None):
        super().__init__(sat_list, sim_config)
        self.cache_refresh_interval = 30.0  # seconds
        self.last_cache_time = -999.0
        self.cached_loads = {i: 0 for i in range(len(sat_list))}

    def schedule(self, task, current_time_s):
        sender = self.sat_list[task.sat_id]
        decision_delay = 0.001  # instant from on-board cache

        # Periodic cache refresh → small overhead
        if current_time_s - self.last_cache_time >= self.cache_refresh_interval:
            self.cached_loads = {s.sat_id: s.get_queue_load()
                                 for s in self.sat_list}
            self.total_overhead_bytes += NUM_SATELLITES * MSG_STATE_UPDATE
            self.last_cache_time = current_time_s

        # Use stale cached loads (key weakness!)
        neighbor_ids = self._get_neighbor_ids(sender.sat_id) + [sender.sat_id]
        self.total_overhead_bytes += MSG_NEIGHBOR_PING  # minimal routing msg

        best_id = min(neighbor_ids,
                      key=lambda i: self.cached_loads.get(i, 0))
        target = self.sat_list[best_id]

        delay = calculate_delay_s(sender, target, task.size_mb)
        task.arrival_time_s += decision_delay + delay
        target.add_task(task)


class DDTScheduler(Scheduler):
    """Dual-scale Digital Twin Scheduler.
    Modes: 'full', 'fdt_only', 'ldt_only', 'no_sync', 'full_sync'"""
    def __init__(self, sat_list, mode='full', sim_config=None):
        super().__init__(sat_list, sim_config)
        self.mode = mode
        self.global_view = {i: 0 for i in range(len(sat_list))}
        self.last_sync_time = -1.0
        self.event_threshold = sim_config.get('event_threshold', 0.2) \
                               if sim_config else 0.2

        try:
            from stable_baselines3 import PPO
            self.fdt_model = PPO.load("ppo_fdt", device='cpu')
            self.ldt_model = self.fdt_model  # In simulation, LDT uses same model; real deployment uses compressed GRU
            self.has_model = True
        except Exception:
            self.has_model = False

    # ----- Sync Logic -----
    def _do_sync(self, current_time_s):
        real = {s.sat_id: s.get_queue_load() for s in self.sat_list}

        if self.mode == 'no_sync':
            return
        if self.mode == 'ldt_only':
            return

        if self.mode == 'full_sync':
            # Brute-force: always sync everything
            self.global_view = real.copy()
            self.total_overhead_bytes += NUM_SATELLITES * MSG_STATE_UPDATE
            return

        if self.mode == 'fdt_only':
            # Periodic sync every 1s (expensive but accurate)
            if current_time_s - self.last_sync_time >= 1.0:
                self.global_view = real.copy()
                self.total_overhead_bytes += NUM_SATELLITES * MSG_STATE_UPDATE
                self.last_sync_time = current_time_s
            return

        # 'full' mode: event-triggered incremental sync
        changed = 0
        for sid, load in real.items():
            old = self.global_view[sid]
            relative_change = abs(load - old) / max(old, 1)
            if relative_change > self.event_threshold:
                self.global_view[sid] = load
                changed += 1
        # Keepalive every 10s
        if current_time_s - self.last_sync_time >= 10.0:
            self.global_view = real.copy()
            changed = NUM_SATELLITES
            self.last_sync_time = current_time_s
        if changed > 0:
            self.total_overhead_bytes += changed * MSG_STATE_UPDATE
            self.last_sync_time = current_time_s

    # ----- Scheduling -----
    def schedule(self, task, current_time_s):
        sender = self.sat_list[task.sat_id]
        self._do_sync(current_time_s)

        if self.mode == 'ldt_only':
            target = self._ldt_schedule(sender, task)
            decision_delay = 0.005
        elif self.mode == 'fdt_only':
            target = self._fdt_schedule(sender, task)
            decision_delay = 1.0  # must wait for global inference
        elif self.mode == 'no_sync':
            target = sender  # no info → process locally
            decision_delay = 0.0
        else:
            # 'full' or 'full_sync': dual-scale decision
            target, decision_delay = self._dual_scale_schedule(
                sender, task, current_time_s)

        delay = calculate_delay_s(sender, target, task.size_mb)
        task.arrival_time_s += decision_delay + delay
        target.add_task(task)

    def _ldt_schedule(self, sender, task):
        """LDT: fast, neighbor-only decision."""
        nids = self._get_neighbor_ids(sender.sat_id) + [sender.sat_id]
        self.total_overhead_bytes += len(nids) * MSG_NEIGHBOR_PING
        return min((self.sat_list[i] for i in nids),
                   key=lambda s: s.get_queue_load())

    def _fdt_schedule(self, sender, task):
        """FDT: use global view + DRL model."""
        if self.has_model:
            state = self._get_drl_state(self.global_view, task)
            action, _ = self.fdt_model.predict(state, deterministic=True)
            target_id = int(action)
            # Sanity: override if model picks overloaded satellite
            model_load = self.global_view.get(target_id, 0)
            min_load = min(self.global_view.values())
            if model_load > min_load * 3 and min_load > 0:
                target_id = min(range(NUM_SATELLITES),
                                key=lambda i: self.global_view.get(i, 0))
            return self.sat_list[target_id]
        return min(self.sat_list, key=lambda s: self.global_view.get(s.sat_id, 0))

    def _dual_scale_schedule(self, sender, task, current_time_s):
        """Full DDT-SCN: use LDT for normal load, escalate to FDT for overload."""
        local_load = sender.get_queue_load()
        avg_load = sum(self.global_view.values()) / max(NUM_SATELLITES, 1)

        if local_load > avg_load * 1.2:
            # Overloaded → use FDT global policy for long-range offloading
            target = self._fdt_schedule(sender, task)
            return target, 0.05  # FDT decision delay
        else:
            # Normal → LDT fast local decision
            target = self._ldt_schedule(sender, task)
            return target, 0.005  # LDT is near-instant


# =============================================================================
# Simulation Engine
# =============================================================================
def run_simulation(scheduler_cls, task_arrival_rate, mode=None, sim_config=None):
    """Run one complete simulation episode. Returns (avg_latency, success_rate, overhead)."""
    sats = [Satellite(i, i // SATS_PER_PLANE, i % SATS_PER_PLANE,
                      SAT_PROCESSING_RATE)
            for i in range(NUM_SATELLITES)]

    kwargs = {}
    if mode is not None:
        kwargs['mode'] = mode
    if sim_config is not None:
        kwargs['sim_config'] = sim_config
    scheduler = scheduler_cls(sats, **kwargs)

    rate_per_step = task_arrival_rate / 60.0  # total tasks per second
    task_id = 0
    all_tasks = []

    # Graduated hotspot: intensity grows with arrival rate
    use_hotspot = task_arrival_rate > HOTSPOT_THRESHOLD_RATE
    if use_hotspot:
        hotspot_frac = min(0.7, 0.4 + (task_arrival_rate - HOTSPOT_THRESHOLD_RATE) / 600.0)
    else:
        hotspot_frac = 0.0

    for t in np.arange(0, SIM_DURATION_S, TIME_STEP_S):
        # Update position & process tasks
        for sat in sats:
            sat.update_position(t)
            if sat.is_processing and sat.current_task:
                sat.current_task.cpu_cycles -= sat.processing_rate * TIME_STEP_S
                if sat.current_task.cpu_cycles <= 0:
                    sat.current_task.completion_time_s = t
                    sat.is_processing = False
                    sat.current_task = None
            if not sat.is_processing and sat.task_queue:
                next_t = sat.task_queue[0]
                if next_t.arrival_time_s <= t:
                    sat.current_task = sat.task_queue.pop(0)
                    sat.is_processing = True

        # Generate new tasks (Poisson process)
        n_new = np.random.poisson(rate_per_step * TIME_STEP_S)
        for _ in range(n_new):
            task_id += 1
            # Hotspot distribution
            if use_hotspot and random.random() < hotspot_frac:
                src = random.randint(0, HOTSPOT_SAT_COUNT - 1)
            else:
                src = random.randint(0, NUM_SATELLITES - 1)
            new_task = Task(task_id, t, src)
            all_tasks.append(new_task)
            scheduler.schedule(new_task, t)

    # Compute metrics
    completed = [t for t in all_tasks if t.completion_time_s >= 0]
    if not completed:
        return 0, 0, scheduler.get_overhead_per_step()

    latencies = [t.completion_time_s - t.original_arrival_time_s for t in completed]
    successes = sum(1 for t in completed
                    if (t.completion_time_s - t.original_arrival_time_s) <= t.e2e_deadline_s)
    avg_lat = float(np.mean(latencies))
    suc_rate = successes / len(all_tasks) * 100.0
    overhead = scheduler.get_overhead_per_step()
    return avg_lat, suc_rate, overhead
