# Panther Lake Simulator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a high-fidelity, modular, cycle-approximate Python simulator of Intel's Panther Lake processor architecture, modeling disaggregated tiles (Compute, Graphics, Platform Controller) interconnected by a Foveros 3D fabric.

**Architecture:** A cycle-approximate simulator modeling the P-cores (Cougar Cove), E-cores/LP-cores (Darkmont), Xe3-LPG GPU, NPU 5, LPDDR5X-9600 controller, SLC, Platform Controller Tile, and Foveros 3D interconnect, driven by JSON-defined workload scenarios.

**Tech Stack:** Python 3.13+, standard library (unittest, typing, json, argparse, time, math, csv).

---

### Task 1: Project Structure Setup

**Files:**
- Create: `panther_lake/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Write the failing test**
N/A (Project structure setup)

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
Write empty `__init__.py` files in `panther_lake` and `tests` directories to define them as Python packages.

**Step 4: Run test to verify it passes**
Run: `python -c "import panther_lake; print('Success')"`
Expected: Prints `Success`

**Step 5: Commit**
```bash
git add panther_lake/__init__.py tests/__init__.py
git commit -m "infra: initialize project package structure"
```

---

### Task 2: Core Execution Models (Cougar Cove & Darkmont)

**Files:**
- Create: `panther_lake/cores.py`
- Create: `tests/test_cores.py`

**Step 1: Write the failing test**
Create `tests/test_cores.py`:
```python
import unittest
from panther_lake.cores import CougarCoveCore, DarkmontCore

class TestCores(unittest.TestCase):
    def test_cougar_cove_execution(self):
        core = CougarCoveCore(core_id=0)
        # Simulate execution of 1000 instructions (80% INT, 20% MEM)
        cycles = core.execute(instructions=1000, mem_ops=200)
        self.assertGreater(cycles, 0)
        self.assertEqual(core.instructions_retired, 1000)

    def test_darkmont_execution(self):
        core = DarkmontCore(core_id=0, is_lp=False)
        cycles = core.execute(instructions=1000, mem_ops=200)
        self.assertGreater(cycles, 0)
        self.assertEqual(core.instructions_retired, 1000)
```

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests/test_cores.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'panther_lake.cores'`

**Step 3: Write minimal implementation**
Create `panther_lake/cores.py`:
```python
import math

class CpuCore:
    def __init__(self, core_id: int, core_type: str, frequency_ghz: float, ipc_target: float):
        self.core_id = core_id
        self.core_type = core_type
        self.frequency_ghz = frequency_ghz
        self.ipc_target = ipc_target
        self.instructions_retired = 0
        self.cycles_elapsed = 0
        self.active_power_watts = 0.0
        self.leakage_power_watts = 0.0
        self.l1_hits = 0
        self.l1_misses = 0

    def get_power(self) -> float:
        return self.active_power_watts + self.leakage_power_watts

class CougarCoveCore(CpuCore):
    """P-Core: Out-of-order execution, higher IPC, higher power."""
    def __init__(self, core_id: int, frequency_ghz: float = 5.0):
        super().__init__(core_id, "P-Core (Cougar Cove)", frequency_ghz, ipc_target=3.0)
        self.leakage_power_watts = 0.5
        
    def execute(self, instructions: int, mem_ops: int) -> int:
        l1_hits = int(mem_ops * 0.95)
        l1_misses = mem_ops - l1_hits
        self.l1_hits += l1_hits
        self.l1_misses += l1_misses
        
        exec_cycles = instructions / self.ipc_target
        mem_penalty_cycles = (l1_hits * 4) + (l1_misses * 15)
        
        total_cycles = int(exec_cycles + (mem_penalty_cycles * 0.4))
        self.cycles_elapsed += total_cycles
        self.instructions_retired += instructions
        
        self.active_power_watts = 4.0 * (self.frequency_ghz / 5.0)**3
        return total_cycles

class DarkmontCore(CpuCore):
    """E-Core: In-order execution, lower IPC, highly efficient."""
    def __init__(self, core_id: int, is_lp: bool = False, frequency_ghz: float = 3.5):
        core_type = "LP E-Core" if is_lp else "E-Core (Darkmont)"
        freq = 2.5 if is_lp else frequency_ghz
        super().__init__(core_id, core_type, freq, ipc_target=1.5 if not is_lp else 1.1)
        self.is_lp = is_lp
        self.leakage_power_watts = 0.05 if is_lp else 0.1
        
    def execute(self, instructions: int, mem_ops: int) -> int:
        l1_hits = int(mem_ops * 0.9)
        l1_misses = mem_ops - l1_hits
        self.l1_hits += l1_hits
        self.l1_misses += l1_misses
        
        exec_cycles = instructions / self.ipc_target
        mem_penalty_cycles = (l1_hits * 4) + (l1_misses * 20)
        
        total_cycles = int(exec_cycles + (mem_penalty_cycles * 0.8))
        self.cycles_elapsed += total_cycles
        self.instructions_retired += instructions
        
        base_power = 0.5 if self.is_lp else 1.2
        self.active_power_watts = base_power * (self.frequency_ghz / 3.0)**3
        return total_cycles
```

**Step 4: Run test to verify it passes**
Run: `python -m unittest tests/test_cores.py`
Expected: PASS

**Step 5: Commit**
```bash
git add panther_lake/cores.py tests/test_cores.py
git commit -m "feat: implement CPU P-core and E-core simulators"
```

---

### Task 3: GPU Execution Model (Xe3-LPG)

**Files:**
- Create: `panther_lake/gpu.py`
- Create: `tests/test_gpu.py`

**Step 1: Write the failing test**
Create `tests/test_gpu.py`:
```python
import unittest
from panther_lake.gpu import Xe3GPU

class TestGPU(unittest.TestCase):
    def test_gpu_simulation(self):
        gpu = Xe3GPU(xe_cores=12, frequency_ghz=2.5)
        cycles = gpu.execute(ops=1_000_000, matrix_ops=50_000)
        self.assertGreater(cycles, 0)
        self.assertEqual(gpu.ops_completed, 1_000_000 + 50_000)
        self.assertGreater(gpu.get_power(), 0.0)
```

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests/test_gpu.py`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**
Create `panther_lake/gpu.py`:
```python
class Xe3GPU:
    """Xe3-LPG Graphics Tile (Celestial)."""
    def __init__(self, xe_cores: int = 12, frequency_ghz: float = 2.5):
        self.xe_cores = xe_cores
        self.frequency_ghz = frequency_ghz
        self.ops_completed = 0
        self.cycles_elapsed = 0
        self.active_power_watts = 0.0
        self.leakage_power_watts = 0.25 * xe_cores
        
        self.vector_engines = xe_cores * 8
        self.matrix_engines = xe_cores * 8
        
        self.ve_flops_per_cycle = 16
        self.xmx_flops_per_cycle = 128

    def execute(self, ops: int, matrix_ops: int) -> int:
        vector_capacity = self.vector_engines * self.ve_flops_per_cycle
        matrix_capacity = self.matrix_engines * self.xmx_flops_per_cycle
        
        ve_cycles = ops / vector_capacity if vector_capacity > 0 else 0
        xmx_cycles = matrix_ops / matrix_capacity if matrix_capacity > 0 else 0
        
        total_cycles = int(max(ve_cycles, xmx_cycles) + 0.8 * min(ve_cycles, xmx_cycles))
        if total_cycles == 0 and (ops > 0 or matrix_ops > 0):
            total_cycles = 1
            
        self.cycles_elapsed += total_cycles
        self.ops_completed += (ops + matrix_ops)
        
        dynamic_power_factor = 2.5 * (self.xe_cores / 12)
        self.active_power_watts = dynamic_power_factor * (self.frequency_ghz / 2.5)**3
        
        return total_cycles

    def get_power(self) -> float:
        return self.active_power_watts + self.leakage_power_watts

    def get_gflops(self) -> float:
        ve_gflops = (self.vector_engines * self.ve_flops_per_cycle * self.frequency_ghz)
        xmx_gflops = (self.matrix_engines * self.xmx_flops_per_cycle * self.frequency_ghz)
        return ve_gflops + xmx_gflops
```

**Step 4: Run test to verify it passes**
Run: `python -m unittest tests/test_gpu.py`
Expected: PASS

**Step 5: Commit**
```bash
git add panther_lake/gpu.py tests/test_gpu.py
git commit -m "feat: implement Xe3-LPG GPU simulation model"
```

---

### Task 4: NPU 5 Execution Model

**Files:**
- Create: `panther_lake/npu.py`
- Create: `tests/test_npu.py`

**Step 1: Write the failing test**
Create `tests/test_npu.py`:
```python
import unittest
from panther_lake.npu import NPU5

class TestNPU(unittest.TestCase):
    def test_npu_inference(self):
        npu = NPU5(frequency_ghz=1.8)
        cycles = npu.execute_matmul(m=128, n=128, k=128)
        self.assertGreater(cycles, 0)
        self.assertEqual(npu.macs_completed, 128 * 128 * 128)
        self.assertGreater(npu.get_power(), 0.0)
        self.assertGreaterEqual(npu.get_max_tops(), 40.0)
```

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests/test_npu.py`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**
Create `panther_lake/npu.py`:
```python
class NPU5:
    """Intel NPU 5 Neural Processing Unit (up to 50 TOPS)."""
    def __init__(self, frequency_ghz: float = 1.8):
        self.frequency_ghz = frequency_ghz
        self.macs_completed = 0
        self.cycles_elapsed = 0
        self.active_power_watts = 0.0
        self.leakage_power_watts = 0.2
        
        self.num_nce = 4
        self.macs_per_nce_cycle = 4096

    def execute_matmul(self, m: int, n: int, k: int) -> int:
        total_macs = m * n * k
        capacity_per_cycle = self.num_nce * self.macs_per_nce_cycle
        
        compute_cycles = total_macs / capacity_per_cycle
        total_cycles = int(compute_cycles * 1.1)
        if total_cycles == 0 and total_macs > 0:
            total_cycles = 1
            
        self.cycles_elapsed += total_cycles
        self.macs_completed += total_macs
        
        utilization = min(1.0, compute_cycles / max(1, total_cycles))
        self.active_power_watts = 8.0 * (self.frequency_ghz / 1.8) * utilization
        
        return total_cycles

    def get_power(self) -> float:
        return self.active_power_watts + self.leakage_power_watts

    def get_max_tops(self) -> float:
        ops_per_cycle = self.num_nce * self.macs_per_nce_cycle * 2
        tops = (ops_per_cycle * self.frequency_ghz) / 1000.0
        return tops
```

**Step 4: Run test to verify it passes**
Run: `python -m unittest tests/test_npu.py`
Expected: PASS

**Step 5: Commit**
```bash
git add panther_lake/npu.py tests/test_npu.py
git commit -m "feat: implement NPU 5 performance and power model"
```

---

### Task 5: Memory Hierarchy and SLC

**Files:**
- Create: `panther_lake/memory.py`
- Create: `tests/test_memory.py`

**Step 1: Write the failing test**
Create `tests/test_memory.py`:
```python
import unittest
from panther_lake.memory import LPDDR5XController, SystemLevelCache

class TestMemory(unittest.TestCase):
    def test_memory_controller(self):
        controller = LPDDR5XController(speed_mts=9600, channels=2)
        latency_ns = controller.access_latency_ns(is_write=False)
        self.assertGreater(latency_ns, 30.0)
        
        bandwidth_gbs = controller.get_peak_bandwidth_gbs()
        self.assertAlmostEqual(bandwidth_gbs, 153.6, places=1)
        
    def test_system_level_cache(self):
        slc = SystemLevelCache(size_mb=24)
        hit, cycles = slc.access(address=0x1000)
        self.assertIn(hit, [True, False])
        self.assertGreater(cycles, 0)
```

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests/test_memory.py`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**
Create `panther_lake/memory.py`:
```python
import random

class LPDDR5XController:
    """LPDDR5X Memory Controller."""
    def __init__(self, speed_mts: float = 9600.0, channels: int = 2):
        self.speed_mts = speed_mts
        self.channels = channels
        self.data_width_bits = 64
        self.reads = 0
        self.writes = 0
        self.bytes_transferred = 0

    def get_peak_bandwidth_gbs(self) -> float:
        width_bytes = self.data_width_bits / 8
        return (self.speed_mts * 10**6 * width_bytes * self.channels) / 10**9

    def access_latency_ns(self, is_write: bool) -> float:
        base = 45.0 if not is_write else 35.0
        queue_delay = random.uniform(5.0, 15.0)
        return base + queue_delay

    def transfer(self, size_bytes: int, is_write: bool):
        if is_write:
            self.writes += 1
        else:
            self.reads += 1
        self.bytes_transferred += size_bytes

class SystemLevelCache:
    """System Level Cache (SLC) shared across compute and graphics tiles."""
    def __init__(self, size_mb: int = 24):
        self.size_mb = size_mb
        self.latency_cycles = 45
        self.hits = 0
        self.misses = 0
        self.cached_lines = set()

    def access(self, address: int) -> tuple[bool, int]:
        line = address >> 6
        if line in self.cached_lines:
            self.hits += 1
            return True, self.latency_cycles
        else:
            self.misses += 1
            if len(self.cached_lines) > (self.size_mb * 1024 * 1024 / 64):
                self.cached_lines.pop()
            self.cached_lines.add(line)
            return False, self.latency_cycles
```

**Step 4: Run test to verify it passes**
Run: `python -m unittest tests/test_memory.py`
Expected: PASS

**Step 5: Commit**
```bash
git add panther_lake/memory.py tests/test_memory.py
git commit -m "feat: implement LPDDR5X and System Level Cache simulator models"
```

---

### Task 6: Foveros 3D Interconnect

**Files:**
- Create: `panther_lake/fabric.py`
- Create: `tests/test_fabric.py`

**Step 1: Write the failing test**
Create `tests/test_fabric.py`:
```python
import unittest
from panther_lake.fabric import FoverosInterconnect

class TestFabric(unittest.TestCase):
    def test_routing_latency(self):
        fabric = FoverosInterconnect()
        latency_cycles = fabric.route(src="ComputeTile", dst="GraphicsTile", data_size_bytes=64)
        self.assertGreater(latency_cycles, 0)
        self.assertEqual(fabric.transfers_count, 1)
        self.assertEqual(fabric.total_bytes_routed, 64)
```

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests/test_fabric.py`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**
Create `panther_lake/fabric.py`:
```python
class FoverosInterconnect:
    """Foveros 3D Packaging Interconnect Fabric."""
    def __init__(self):
        self.transfers_count = 0
        self.total_bytes_routed = 0
        self.contention_count = 0
        
        self.links = {
            ("ComputeTile", "GraphicsTile"): 256.0,
            ("ComputeTile", "PlatformControllerTile"): 128.0,
            ("GraphicsTile", "PlatformControllerTile"): 64.0
        }
        self.base_latencies = {
            ("ComputeTile", "GraphicsTile"): 8,
            ("ComputeTile", "PlatformControllerTile"): 12,
            ("GraphicsTile", "PlatformControllerTile"): 16
        }

    def route(self, src: str, dst: str, data_size_bytes: int) -> int:
        self.transfers_count += 1
        self.total_bytes_routed += data_size_bytes
        
        key = (src, dst) if (src, dst) in self.links else (dst, src)
        
        if key not in self.links:
            return 1
            
        base_lat = self.base_latencies[key]
        bandwidth = self.links[key]
        
        transfer_ns = (data_size_bytes / (bandwidth * 10**9)) * 10**9
        transfer_cycles = int(transfer_ns / 0.5)
        
        contention_cycles = 0
        if self.transfers_count % 100 == 0:
            self.contention_count += 1
            contention_cycles = 5
            
        total_latency = base_lat + transfer_cycles + contention_cycles
        return total_latency
```

**Step 4: Run test to verify it passes**
Run: `python -m unittest tests/test_fabric.py`
Expected: PASS

**Step 5: Commit**
```bash
git add panther_lake/fabric.py tests/test_fabric.py
git commit -m "feat: implement Foveros 3D Interconnect routing model"
```

---

### Task 7: Platform Controller Tile

**Files:**
- Create: `panther_lake/platform.py`
- Create: `tests/test_platform.py`

**Step 1: Write the failing test**
Create `tests/test_platform.py`:
```python
import unittest
from panther_lake.platform import PlatformControllerTile

class TestPlatform(unittest.TestCase):
    def test_platform_io_and_dvfs(self):
        pct = PlatformControllerTile()
        cycles = pct.simulate_io(pcie_traffic_bytes=1000000, tb_traffic_bytes=500000)
        self.assertGreater(cycles, 0)
        self.assertGreater(pct.get_power(), 0.0)
        
        voltage = pct.calculate_dvfs_voltage(freq_ghz=5.0)
        self.assertGreater(voltage, 0.7)
```

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests/test_platform.py`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**
Create `panther_lake/platform.py`:
```python
class PlatformControllerTile:
    """Platform Controller Tile (PCT) built on TSMC N6."""
    def __init__(self):
        self.pcie_traffic = 0
        self.tb_traffic = 0
        self.active_power_watts = 0.5
        self.leakage_power_watts = 0.3
        
        self.pcie_max_gbs = 63.0
        self.tb_max_gbs = 10.0

    def simulate_io(self, pcie_traffic_bytes: int, tb_traffic_bytes: int) -> int:
        self.pcie_traffic += pcie_traffic_bytes
        self.tb_traffic += tb_traffic_bytes
        
        pcie_transfer_ns = (pcie_traffic_bytes / (self.pcie_max_gbs * 10**9)) * 10**9
        tb_transfer_ns = (tb_traffic_bytes / (self.tb_max_gbs * 10**9)) * 10**9
        
        pct_cycles = int(max(pcie_transfer_ns, tb_transfer_ns))
        
        total_traffic = pcie_traffic_bytes + tb_traffic_bytes
        io_utilization = min(1.0, total_traffic / 10**7)
        self.active_power_watts = 0.5 + 2.0 * io_utilization
        
        return max(1, pct_cycles)

    def calculate_dvfs_voltage(self, freq_ghz: float) -> float:
        if freq_ghz <= 1.0:
            return 0.70
        elif freq_ghz >= 6.0:
            return 1.35
        ratio = (freq_ghz - 1.0) / 5.0
        return 0.70 + ratio * (1.35 - 0.70)

    def get_power(self) -> float:
        return self.active_power_watts + self.leakage_power_watts
```

**Step 4: Run test to verify it passes**
Run: `python -m unittest tests/test_platform.py`
Expected: PASS

**Step 5: Commit**
```bash
git add panther_lake/platform.py tests/test_platform.py
git commit -m "feat: implement Platform Controller Tile (PCT) simulator model"
```

---

### Task 8: Simulator Orchestrator

**Files:**
- Create: `panther_lake/simulator.py`
- Create: `tests/test_simulator.py`

**Step 1: Write the failing test**
Create `tests/test_simulator.py`:
```python
import unittest
from panther_lake.simulator import PantherLakeProcessor, SimulationResult

class TestSimulator(unittest.TestCase):
    def test_simulation_run(self):
        processor = PantherLakeProcessor(high_power_config=True)
        workload = {
            "cpu_instructions": 500_000,
            "cpu_mem_ops": 100_000,
            "gpu_ops": 100_000,
            "gpu_matrix_ops": 20_000,
            "npu_m": 128, "npu_n": 128, "npu_k": 128,
            "pcie_bytes": 50_000,
            "tb_bytes": 10_000
        }
        res = processor.run_workload(workload)
        self.assertIsInstance(res, SimulationResult)
        self.assertGreater(res.total_cycles, 0)
        self.assertGreater(res.avg_power_watts, 0)
        self.assertGreater(res.total_energy_joules, 0)
```

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests/test_simulator.py`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**
Create `panther_lake/simulator.py`:
```python
import random
from typing import Dict, Any
from panther_lake.cores import CougarCoveCore, DarkmontCore
from panther_lake.gpu import Xe3GPU
from panther_lake.npu import NPU5
from panther_lake.memory import LPDDR5XController, SystemLevelCache
from panther_lake.fabric import FoverosInterconnect
from panther_lake.platform import PlatformControllerTile

class SimulationResult:
    def __init__(self, total_cycles: int, simulation_time_sec: float, avg_power_watts: float,
                 total_energy_joules: float, cpu_ips: float, gpu_gflops: float, npu_tops: float,
                 peak_temp_c: float):
        self.total_cycles = total_cycles
        self.simulation_time_sec = simulation_time_sec
        self.avg_power_watts = avg_power_watts
        self.total_energy_joules = total_energy_joules
        self.cpu_ips = cpu_ips
        self.gpu_gflops = gpu_gflops
        self.npu_tops = npu_tops
        self.peak_temp_c = peak_temp_c

class PantherLakeProcessor:
    """Orchestrates all Panther Lake Tiles and runs simulation cycles."""
    def __init__(self, high_power_config: bool = True):
        self.high_power_config = high_power_config
        self.p_cores = [CougarCoveCore(core_id=i, frequency_ghz=5.0) for i in range(4)]
        
        if high_power_config:
            self.e_cores = [DarkmontCore(core_id=i, is_lp=False, frequency_ghz=3.5) for i in range(8)]
            self.lp_cores = [DarkmontCore(core_id=i, is_lp=True, frequency_ghz=2.5) for i in range(4)]
            self.gpu = Xe3GPU(xe_cores=12, frequency_ghz=2.5)
        else:
            self.e_cores = []
            self.lp_cores = [DarkmontCore(core_id=i, is_lp=True, frequency_ghz=2.5) for i in range(4)]
            self.gpu = Xe3GPU(xe_cores=4, frequency_ghz=2.0)
            
        self.npu = NPU5(frequency_ghz=1.8)
        self.memory_ctrl = LPDDR5XController(speed_mts=9600.0, channels=2)
        self.slc = SystemLevelCache(size_mb=24 if high_power_config else 16)
        self.fabric = FoverosInterconnect()
        self.pct = PlatformControllerTile()

    def run_workload(self, workload: Dict[str, Any]) -> SimulationResult:
        cpu_insts = workload.get("cpu_instructions", 0)
        cpu_mem_ops = workload.get("cpu_mem_ops", 0)
        gpu_ops = workload.get("gpu_ops", 0)
        gpu_matrix_ops = workload.get("gpu_matrix_ops", 0)
        npu_m = workload.get("npu_m", 0)
        npu_n = workload.get("npu_n", 0)
        npu_k = workload.get("npu_k", 0)
        pcie_bytes = workload.get("pcie_bytes", 0)
        tb_bytes = workload.get("tb_bytes", 0)
        
        all_cpu_cores = self.p_cores + self.e_cores + self.lp_cores
        
        cpu_cycles = 0
        p_work = int(cpu_insts * 0.7 / len(self.p_cores))
        p_mem = int(cpu_mem_ops * 0.7 / len(self.p_cores))
        for core in self.p_cores:
            cpu_cycles = max(cpu_cycles, core.execute(p_work, p_mem))
            
        if self.e_cores:
            e_work = int(cpu_insts * 0.25 / len(self.e_cores))
            e_mem = int(cpu_mem_ops * 0.25 / len(self.e_cores))
            for core in self.e_cores:
                cpu_cycles = max(cpu_cycles, core.execute(e_work, e_mem))
                
            lp_work = int(cpu_insts * 0.05 / len(self.lp_cores))
            lp_mem = int(cpu_mem_ops * 0.05 / len(self.lp_cores))
            for core in self.lp_cores:
                cpu_cycles = max(cpu_cycles, core.execute(lp_work, lp_mem))
        else:
            lp_work = int(cpu_insts * 0.3 / len(self.lp_cores))
            lp_mem = int(cpu_mem_ops * 0.3 / len(self.lp_cores))
            for core in self.lp_cores:
                cpu_cycles = max(cpu_cycles, core.execute(lp_work, lp_mem))
                
        gpu_cycles = self.gpu.execute(gpu_ops, gpu_matrix_ops)
        npu_cycles = self.npu.execute_matmul(npu_m, npu_n, npu_k)
        pct_cycles = self.pct.simulate_io(pcie_bytes, tb_bytes)
        
        cpu_misses = sum(c.l1_misses for c in all_cpu_cores)
        slc_access_cycles = 0
        
        for _ in range(cpu_misses):
            addr = hash(random.random())
            hit, latency = self.slc.access(addr)
            slc_access_cycles += latency
            if not hit:
                self.memory_ctrl.transfer(64, is_write=False)
                
        fabric_cycles = self.fabric.route("ComputeTile", "GraphicsTile", gpu_ops * 4) + \
                        self.fabric.route("ComputeTile", "PlatformControllerTile", pcie_bytes + tb_bytes)
                        
        total_cycles = max(cpu_cycles, gpu_cycles, npu_cycles, pct_cycles) + fabric_cycles + slc_access_cycles
        total_power = sum(c.get_power() for c in all_cpu_cores) + self.gpu.get_power() + self.npu.get_power() + self.pct.get_power()
        
        simulation_time_sec = total_cycles / (2.0 * 10**9)
        total_energy_joules = total_power * simulation_time_sec
        
        cpu_ips = cpu_insts / simulation_time_sec if simulation_time_sec > 0 else 0
        gpu_gflops = (gpu_ops + gpu_matrix_ops * 2) / (simulation_time_sec * 10**9) if simulation_time_sec > 0 else 0
        npu_tops = (npu_m * npu_n * npu_k * 2) / (simulation_time_sec * 10**12) if simulation_time_sec > 0 else 0
        
        peak_temp_c = 35.0 + 1.2 * total_power
        
        return SimulationResult(
            total_cycles=total_cycles,
            simulation_time_sec=simulation_time_sec,
            avg_power_watts=total_power,
            total_energy_joules=total_energy_joules,
            cpu_ips=cpu_ips,
            gpu_gflops=gpu_gflops,
            npu_tops=npu_tops,
            peak_temp_c=peak_temp_c
        )
```

**Step 4: Run test to verify it passes**
Run: `python -m unittest tests/test_simulator.py`
Expected: PASS

**Step 5: Commit**
```bash
git add panther_lake/simulator.py tests/test_simulator.py
git commit -m "feat: implement full PantherLakeProcessor coordinator and simulator"
```

---

### Task 9: Simulator CLI and Workloads

**Files:**
- Create: `workloads/ai_inference.json`
- Create: `workloads/triple_a_gaming.json`
- Create: `workloads/code_compilation.json`
- Create: `simulate.py`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
Write the JSON files and the `simulate.py` script as specified in the simulator spec.

**Step 4: Run test to verify it passes**
Run: `python simulate.py --workload workloads/ai_inference.json --config high_power`
Expected: Outputs statistics successfully.

**Step 5: Commit**
```bash
git add workloads/ simulate.py
git commit -m "feat: add simulator CLI and standard workloads"
```

---

### Task 10: DESIGN.md Documentation

**Files:**
- Create: `DESIGN.md`
- Create: `README.md`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
Create a detailed layout explanation in `DESIGN.md` and standard setup details in `README.md`.

**Step 4: Run test to verify it passes**
N/A

**Step 5: Commit**
```bash
git add DESIGN.md README.md
git commit -m "docs: write architectural design and README documentation"
```
