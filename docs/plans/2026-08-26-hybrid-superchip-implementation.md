# Hybrid AI Superchip Simulator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the custom Intel-NVIDIA Hybrid AI Superchip simulator in C++17, modeling polymorphic execution cores (`ExecutionCore`), dynamic memory bandwidth allocation, and a strict 35W power limit (DVFS) running a 40B parameter FP16 LLM workload.

**Architecture:** Refactor `cores.hpp` to introduce a unified `ExecutionCore` class. Create the `BlackwellGpuCore` class. Refactor `simulator.cpp` to enforce dynamic memory partitioning (85% priority shift based on workload phase) and a 35W power cap (DVFS throttling GPU/CPU frequencies).

**Tech Stack:** C++17, CMake.

---

### Task 1: Refactor Base Execution Core Class

**Files:**
- Modify: `include/panther_lake/cores.hpp`
- Modify: `src/cores.cpp`
- Modify: `tests/test_runner.cpp`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`, add tests verifying the base core properties and compile-time compatibility of a shared base class.
```cpp
void test_execution_core_polymorphism() {
    // Verify that we can manage P-cores and E-cores using a unified ExecutionCore pointer
    std::vector<std::unique_ptr<panther_lake::ExecutionCore>> cores;
    cores.push_back(std::make_unique<panther_lake::CougarCoveCore>(0));
    cores.push_back(std::make_unique<panther_lake::DarkmontCore>(1, false));
    
    assert(cores[0]->get_frequency_ghz() == 5.0);
    assert(cores[1]->get_frequency_ghz() == 3.5);
    std::cout << "test_execution_core_polymorphism PASS" << std::endl;
}
```

**Step 2: Run test to verify it fails**
Run: `cmake --build build`
Expected: Compilation fails due to `ExecutionCore` being undefined.

**Step 3: Write minimal implementation**
Modify `include/panther_lake/cores.hpp`:
```cpp
#pragma once
#include <string>

namespace panther_lake {

class ExecutionCore {
public:
    ExecutionCore(int core_id, std::string type, double frequency_ghz, double ipc_target);
    virtual ~ExecutionCore() = default;
    
    virtual unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) = 0;
    double get_power() const;
    double get_frequency_ghz() const { return frequency_ghz_; }
    void set_frequency_ghz(double freq) { frequency_ghz_ = freq; }
    unsigned long long get_retired_instructions() const { return instructions_retired_; }
    unsigned long long get_cycles_elapsed() const { return cycles_elapsed_; }
    unsigned long long get_l1_hits() const { return l1_hits_; }
    unsigned long long get_l1_misses() const { return l1_misses_; }

protected:
    int core_id_;
    std::string type_;
    double frequency_ghz_;
    double ipc_target_;
    unsigned long long instructions_retired_ = 0;
    unsigned long long cycles_elapsed_ = 0;
    double active_power_watts_ = 0.0;
    double leakage_power_watts_ = 0.0;
    unsigned long long l1_hits_ = 0;
    unsigned long long l1_misses_ = 0;
};

class CougarCoveCore : public ExecutionCore {
public:
    CougarCoveCore(int core_id, double frequency_ghz = 5.0);
    unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) override;
};

class DarkmontCore : public ExecutionCore {
public:
    DarkmontCore(int core_id, bool is_lp, double frequency_ghz = 3.5);
    unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) override;
private:
    bool is_lp_;
};

} // namespace panther_lake
```

Modify `src/cores.cpp` constructor signatures to reference `ExecutionCore` instead of `CpuCore`.
Rename references of `CpuCore` in `src/cores.cpp` to `ExecutionCore`.

**Step 4: Run test to verify it passes**
Run: `cmake --build build && ./build/test_runner`
Expected: PASS

**Step 5: Commit**
```bash
git add include/panther_lake/cores.hpp src/cores.cpp tests/test_runner.cpp
git commit -m "feat: refactor core base class to ExecutionCore"
```

---

### Task 2: Implement Blackwell GPU Core

**Files:**
- Modify: `include/panther_lake/cores.hpp`
- Modify: `src/cores.cpp`
- Modify: `tests/test_runner.cpp`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`:
```cpp
void test_blackwell_gpu_core() {
    panther_lake::BlackwellGpuCore gpu_core(0, 1.6);
    unsigned long long cycles = gpu_core.execute(100000, 20000);
    assert(cycles > 0);
    assert(gpu_core.get_power() > 0.0);
    std::cout << "test_blackwell_gpu_core PASS" << std::endl;
}
```

**Step 2: Run test to verify it fails**
Run: `cmake --build build`
Expected: Compilation fails due to `BlackwellGpuCore` undefined.

**Step 3: Write minimal implementation**
Add `BlackwellGpuCore` class inside `include/panther_lake/cores.hpp`:
```cpp
class BlackwellGpuCore : public ExecutionCore {
public:
    BlackwellGpuCore(int core_id, double frequency_ghz = 1.6);
    unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) override;
    double get_vector_flops_per_cycle() const { return 512.0; } // 512 CUDA cores equivalent per SM/cluster
    double get_matrix_flops_per_cycle() const { return 1024.0; } // 8 Tensor cores
};
```

Implement `BlackwellGpuCore` inside `src/cores.cpp`:
```cpp
BlackwellGpuCore::BlackwellGpuCore(int core_id, double frequency_ghz)
    : ExecutionCore(core_id, "Blackwell GPU Core", frequency_ghz, 64.0) {
    leakage_power_watts_ = 0.15;
}

unsigned long long BlackwellGpuCore::execute(unsigned long long instructions, unsigned long long mem_ops) {
    unsigned long long hits = static_cast<unsigned long long>(mem_ops * 0.98);
    unsigned long long misses = mem_ops - hits;
    l1_hits_ += hits;
    l1_misses_ += misses;
    
    double exec_cycles = instructions / ipc_target_;
    double mem_penalty = (hits * 2) + (misses * 20); // GPU has high bandwidth memory L2 cache hit path
    unsigned long long total_cycles = static_cast<unsigned long long>(exec_cycles + (mem_penalty * 0.2));
    
    cycles_elapsed_ += total_cycles;
    instructions_retired_ += instructions;
    
    // Blackwell core dynamic power: scales with frequency cubed
    active_power_watts_ = 2.5 * std::pow(frequency_ghz_ / 1.6, 3);
    
    return total_cycles;
}
```

**Step 4: Run test to verify it passes**
Run: `cmake --build build && ./build/test_runner`
Expected: PASS

**Step 5: Commit**
```bash
git add include/panther_lake/cores.hpp src/cores.cpp tests/test_runner.cpp
git commit -m "feat: implement Blackwell GPU core class"
```

---

### Task 3: Refactor GPU Tile manager

**Files:**
- Modify: `include/panther_lake/gpu.hpp`
- Modify: `src/gpu.cpp`
- Modify: `tests/test_runner.cpp`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`, update `test_gpu()` to verify that the GPU manages Blackwell cores:
```cpp
void test_gpu() {
    panther_lake::Xe3GPU gpu(12, 1.6); // 12 Blackwell GPU cores
    unsigned long long cycles = gpu.execute(1000000, 50000);
    assert(cycles > 0);
    assert(gpu.get_power() > 0.0);
    std::cout << "test_gpu PASS" << std::endl;
}
```

**Step 2: Run test to verify it fails**
Run: `cmake --build build`
Expected: Compilation errors due to old interface mappings in `gpu.cpp`.

**Step 3: Write minimal implementation**
Refactor `include/panther_lake/gpu.hpp`:
```cpp
#pragma once
#include <vector>
#include <memory>
#include "panther_lake/cores.hpp"

namespace panther_lake {

class Xe3GPU {
public:
    Xe3GPU(int xe_cores = 12, double frequency_ghz = 1.6);
    unsigned long long execute(unsigned long long ops, unsigned long long matrix_ops);
    double get_power() const;
    double get_gflops() const;
    int get_xe_cores() const { return xe_cores_; }
    std::vector<std::unique_ptr<BlackwellGpuCore>>& get_gpu_cores() { return gpu_cores_; }

private:
    int xe_cores_;
    double frequency_ghz_;
    std::vector<std::unique_ptr<BlackwellGpuCore>> gpu_cores_;
};

} // namespace panther_lake
```

Refactor `src/gpu.cpp`:
```cpp
#include "panther_lake/gpu.hpp"
#include <algorithm>

namespace panther_lake {

Xe3GPU::Xe3GPU(int xe_cores, double frequency_ghz)
    : xe_cores_(xe_cores), frequency_ghz_(frequency_ghz) {
    for (int i = 0; i < xe_cores_; ++i) {
        gpu_cores_.push_back(std::make_unique<BlackwellGpuCore>(i, frequency_ghz_));
    }
}

unsigned long long Xe3GPU::execute(unsigned long long ops, unsigned long long matrix_ops) {
    unsigned long long max_cycles = 0;
    
    // Distribute ops evenly across GPU cores
    unsigned long long ops_per_core = ops / xe_cores_;
    unsigned long long matrix_ops_per_core = matrix_ops / xe_cores_;
    
    for (auto& core : gpu_cores_) {
        // CUDA core throughput = 512, Tensor core = 1024
        unsigned long long instructions = (ops_per_core / 512) + (matrix_ops_per_core / 1024);
        unsigned long long cycles = core->execute(instructions, (ops_per_core + matrix_ops_per_core) / 64);
        max_cycles = std::max(max_cycles, cycles);
    }
    
    return max_cycles;
}

double Xe3GPU::get_power() const {
    double total = 0.0;
    for (const auto& core : gpu_cores_) {
        total += core->get_power();
    }
    return total;
}

double Xe3GPU::get_gflops() const {
    double total_flops = 0.0;
    for (const auto& core : gpu_cores_) {
        total_flops += (core->get_vector_flops_per_cycle() + core->get_matrix_flops_per_cycle()) * frequency_ghz_;
    }
    return total_flops;
}

} // namespace panther_lake
```

**Step 4: Run test to verify it passes**
Run: `cmake --build build && ./build/test_runner`
Expected: PASS

**Step 5: Commit**
```bash
git add include/panther_lake/gpu.hpp src/gpu.cpp tests/test_runner.cpp
git commit -m "feat: refactor GPU tile to manage Blackwell core instances"
```

---

### Task 4: Dynamic Bandwidth & Power Throttling (Orchestrator)

**Files:**
- Modify: `include/panther_lake/simulator.hpp`
- Modify: `src/simulator.cpp`
- Modify: `tests/test_runner.cpp`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`:
```cpp
void test_dynamic_bandwidth_and_power_limits() {
    panther_lake::PantherLakeProcessor processor(true); // High power config
    panther_lake::Workload workload;
    workload.cpu_instructions = 10000000ULL;
    workload.cpu_mem_ops = 2000000ULL;
    workload.gpu_ops = 800000000ULL;
    workload.gpu_matrix_ops = 200000000ULL;
    workload.npu_m = 2048;
    workload.npu_n = 2048;
    workload.npu_k = 2048;
    
    auto res = processor.run_workload(workload);
    
    // Assert SoC power is capped at 35W
    assert(res.avg_power_watts <= 35.5);
    std::cout << "test_dynamic_bandwidth_and_power_limits PASS" << std::endl;
}
```

**Step 2: Run test to verify it fails**
Run: `cmake --build build && ./build/test_runner`
Expected: Fails or throws assertion failure because power goes over 35W.

**Step 3: Write minimal implementation**
Refactor `run_workload` inside `src/simulator.cpp`:
- Implement dynamic memory bandwidth partitioning:
  - If NPU workload is high (e.g., `workload.npu_m > 0`), route 85% bandwidth to NPU, leaving 15% to CPU/GPU.
  - If GPU workload is high and NPU is low, allocate 85% to GPU.
- Implement TDP 35W Power Cap DVFS Throttling loop:
  - Calculate initial power of all tiles.
  - While total power > 35.0 Watts and frequency of GPU and CPU is above minimum (GPU min 0.8 GHz, P-core min 1.5 GHz, E-core min 1.0 GHz):
    - Reduce GPU core frequencies by 0.1 GHz.
    - Reduce CPU P-core frequencies by 0.1 GHz and E-core frequencies by 0.1 GHz.
    - Recalculate dynamic power.

`src/simulator.cpp` (refactored `run_workload` body snippet):
```cpp
SimulationResult PantherLakeProcessor::run_workload(const Workload& workload) {
    // 1. Dynamic Bandwidth Partitioning setup
    double gpu_bw_fraction = 0.15;
    double npu_bw_fraction = 0.15;
    
    // Determine active phase based on matrix workload versus graphics ops
    bool npu_active = (workload.npu_m > 0 && workload.npu_n > 0 && workload.npu_k > 0);
    bool gpu_active = (workload.gpu_ops > 0 || workload.gpu_matrix_ops > 0);
    
    if (npu_active && !gpu_active) {
        npu_bw_fraction = 0.85;
        gpu_bw_fraction = 0.05;
    } else if (gpu_active && !npu_active) {
        gpu_bw_fraction = 0.85;
        npu_bw_fraction = 0.05;
    } else if (gpu_active && npu_active) {
        // Contention case: dynamically share (GPU token gen gets priority, NPU gets prompt priority)
        gpu_bw_fraction = 0.60;
        npu_bw_fraction = 0.30;
    }
    
    // Enforce memory bandwidth limitations on latency calculations
    double peak_bw = memory_ctrl_.get_peak_bandwidth_gbs();
    double gpu_allocated_bw = peak_bw * gpu_bw_fraction;
    double npu_allocated_bw = peak_bw * npu_bw_fraction;

    // 2. Initial power calculation and DVFS throttling loop
    double target_tdp = 35.0; // 35W total package power cap
    
    double initial_gpu_freq = gpu_.get_gpu_cores()[0]->get_frequency_ghz();
    double initial_p_freq = p_cores_[0]->get_frequency_ghz();
    double initial_e_freq = e_cores_.empty() ? 0.0 : e_cores_[0]->get_frequency_ghz();
    
    double gpu_freq = initial_gpu_freq;
    double p_freq = initial_p_freq;
    double e_freq = initial_e_freq;
    
    while (true) {
        double cpu_power = 0.0;
        for (auto& core : p_cores_) cpu_power += core->get_power();
        for (auto& core : e_cores_) cpu_power += core->get_power();
        for (auto& core : lp_cores_) cpu_power += core->get_power();
        
        double gpu_power = gpu_.get_power();
        double npu_power = npu_.get_power();
        double pct_power = pct_.get_power();
        
        double total_power = cpu_power + gpu_power + npu_power + pct_power;
        
        if (total_power <= target_tdp || (gpu_freq <= 0.8 && p_freq <= 1.5)) {
            break;
        }
        
        // Scale down frequencies (DVFS)
        if (gpu_freq > 0.8) {
            gpu_freq -= 0.1;
            for (auto& core : gpu_.get_gpu_cores()) core->set_frequency_ghz(gpu_freq);
        }
        if (p_freq > 1.5) {
            p_freq -= 0.1;
            for (auto& core : p_cores_) core->set_frequency_ghz(p_freq);
        }
        if (e_freq > 1.0) {
            e_freq -= 0.1;
            for (auto& core : e_cores_) core->set_frequency_ghz(e_freq);
        }
    }
    
    // Enforce bandwidth throttling on execution latencies
    // Latency = Ops / Bandwidth
    unsigned long long gpu_mem_transfer_cycles = static_cast<unsigned long long>(
        (workload.gpu_ops * 4) / (gpu_allocated_bw * 1000000000.0) / 0.5
    );
    unsigned long long npu_mem_transfer_cycles = static_cast<unsigned long long>(
        (static_cast<double>(workload.npu_m) * workload.npu_n * workload.npu_k) / (npu_allocated_bw * 1000000000.0) / 0.5
    );
    
    // Simulate cpu, gpu, npu cycles
    unsigned long long cpu_cycles = 0;
    unsigned long long p_work = static_cast<unsigned long long>(workload.cpu_instructions * 0.7 / p_cores_.size());
    unsigned long long p_mem = static_cast<unsigned long long>(workload.cpu_mem_ops * 0.7 / p_cores_.size());
    for (auto& core : p_cores_) {
        cpu_cycles = std::max(cpu_cycles, core->execute(p_work, p_mem));
    }
    
    if (high_power_config_) {
        unsigned long long e_work = static_cast<unsigned long long>(workload.cpu_instructions * 0.25 / e_cores_.size());
        unsigned long long e_mem = static_cast<unsigned long long>(workload.cpu_mem_ops * 0.25 / e_cores_.size());
        for (auto& core : e_cores_) {
            cpu_cycles = std::max(cpu_cycles, core->execute(e_work, e_mem));
        }
    }
    
    unsigned long long gpu_cycles = gpu_.execute(workload.gpu_ops, workload.gpu_matrix_ops) + gpu_mem_transfer_cycles;
    unsigned long long npu_cycles = npu_.execute_matmul(workload.npu_m, workload.npu_n, workload.npu_k) + npu_mem_transfer_cycles;
    unsigned long long pct_cycles = pct_.simulate_io(workload.pcie_bytes, workload.tb_bytes);
    
    int fabric_cycles = fabric_.route("ComputeTile", "GraphicsTile", workload.gpu_ops * 4) +
                        fabric_.route("ComputeTile", "PlatformControllerTile", workload.pcie_bytes + workload.tb_bytes);
                        
    unsigned long long total_cycles = std::max({cpu_cycles, gpu_cycles, npu_cycles, pct_cycles}) + fabric_cycles;
    
    double final_power = 0.0;
    for (auto& core : p_cores_) final_power += core->get_power();
    for (auto& core : e_cores_) final_power += core->get_power();
    for (auto& core : lp_cores_) final_power += core->get_power();
    final_power += gpu_.get_power() + npu_.get_power() + pct_.get_power();
    
    double simulation_time_sec = total_cycles / (2.0 * 1000000000.0);
    double total_energy_joules = final_power * simulation_time_sec;
    
    double cpu_ips = workload.cpu_instructions / simulation_time_sec;
    double gpu_gflops = (workload.gpu_ops + workload.gpu_matrix_ops * 2) / (simulation_time_sec * 1000000000.0);
    double npu_tops = (static_cast<double>(workload.npu_m) * workload.npu_n * workload.npu_k * 2) / (simulation_time_sec * 1000000000000.0);
    
    double peak_temp_c = 35.0 + 1.2 * final_power;
    
    // Restore base frequencies for next run
    for (auto& core : p_cores_) core->set_frequency_ghz(initial_p_freq);
    for (auto& core : e_cores_) core->set_frequency_ghz(initial_e_freq);
    for (auto& core : gpu_.get_gpu_cores()) core->set_frequency_ghz(initial_gpu_freq);
    
    return SimulationResult{
        total_cycles,
        simulation_time_sec,
        final_power,
        total_energy_joules,
        cpu_ips,
        gpu_gflops,
        npu_tops,
        peak_temp_c
    };
}
```

**Step 4: Run test to verify it passes**
Run: `cmake --build build && ./build/test_runner`
Expected: PASS

**Step 5: Commit**
```bash
git add include/panther_lake/simulator.hpp src/simulator.cpp tests/test_runner.cpp
git commit -m "feat: implement DVFS 35W throttling and dynamic memory partitioning"
```

---

### Task 5: Configure 40B LLM Workload CLI & Dashboard

**Files:**
- Modify: `main.cpp`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
Configure `main.cpp` to feed a 40 Billion parameter model footprint (80 GB transfer) into the simulation workloads, showing prompt processing (NPU-bound) and token generation (GPU-bound) in the console live dashboard.

**Step 4: Run test to verify it passes**
Run: `cmake --build build && ./build/panther_lake_sim`
Expected: Telemetry updates live, showing the correct power metrics (clamped at 35W) and 80 GB parameter throughput calculations.

**Step 5: Commit**
```bash
git add main.cpp
git commit -m "feat: configure main application for 40B LLM oneAPI workload simulation"
```
