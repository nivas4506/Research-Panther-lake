# Panther Lake C++ oneAPI Simulator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a high-fidelity, modular, cycle-approximate C++ simulator of Intel's Panther Lake processor architecture, modeling disaggregated tiles interconnected by a Foveros 3D fabric, and simulating the dispatch and execution of oneAPI (SYCL) kernels on the CPU P/E-cores and the Xe3 GPU.

**Architecture:** The simulator will model a mock `sycl` namespace representing `sycl::queue`, `sycl::device`, `sycl::buffer`, and `sycl::accessor` to allow loading and running actual SYCL-like program queues on a model of Panther Lake hardware (Cougar Cove P-cores, Darkmont E-cores, Xe3 Celestial GPU, NPU 5, Foveros Fabric, and LPDDR5X-9600 memory controller).

**Tech Stack:** C++17, CMake, standard C++ compiler.

---

### Task 1: Project Setup (CMake)

**Files:**
- Create: `CMakeLists.txt`
- Create: `include/panther_lake/sycl_mock.hpp`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
Create a CMake file that sets up an executable and target directories.
Create a blank `sycl_mock.hpp` file under `include/panther_lake/`.

`CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.15)
project(panther_lake_simulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_
REQUIRED ON)

include_directories(include)

add_executable(panther_lake_sim main.cpp)
```

Create an empty `main.cpp` in the root:
```cpp
int main() {
    return 0;
}
```

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build`
Expected: Builds the compiler target successfully.

**Step 5: Commit**
```bash
git add CMakeLists.txt main.cpp include/panther_lake/sycl_mock.hpp
git commit -m "infra: setup CMake and project structure"
```

---

### Task 2: Mock SYCL Namespace (oneAPI interface)

**Files:**
- Modify: `include/panther_lake/sycl_mock.hpp`
- Create: `tests/test_runner.cpp`

**Step 1: Write the failing test**
Create `tests/test_runner.cpp` to verify mock SYCL interfaces.
```cpp
#include <iostream>
#include <cassert>
#include "panther_lake/sycl_mock.hpp"

void test_sycl_mock() {
    sycl::device cpu_dev(sycl::device_type::cpu);
    sycl::device gpu_dev(sycl::device_type::gpu);
    sycl::queue q(gpu_dev);
    
    assert(q.get_device().get_type() == sycl::device_type::gpu);
    std::cout << "test_sycl_mock PASS" << std::endl;
}

int main() {
    test_sycl_mock();
    return 0;
}
```

Add tests to CMakeLists.txt:
```cmake
add_executable(test_runner tests/test_runner.cpp)
```

**Step 2: Run test to verify it fails**
Run: `cmake --build build`
Expected: Compilation failure because `sycl_mock.hpp` is empty.

**Step 3: Write minimal implementation**
Implement `sycl_mock.hpp`:
```cpp
#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace sycl {

enum class device_type {
    cpu,
    gpu,
    npu
};

class device {
public:
    device(device_type t) : type_(t) {}
    device_type get_type() const { return type_; }
    std::string get_name() const {
        switch(type_) {
            case device_type::cpu: return "Cougar Cove / Darkmont Hybrid CPU";
            case device_type::gpu: return "Xe3-LPG Celestial GPU";
            case device_type::npu: return "Intel NPU 5 Accelerator";
        }
        return "Unknown Device";
    }
private:
    device_type type_;
};

class handler {
public:
    template <typename KernelName, typename Func>
    void parallel_for(size_t range, Func f) {
        // Mock kernel registration
        registered_kernels_.push_back([f, range]() {
            for (size_t i = 0; i < range; ++i) {
                f(i);
            }
        });
    }
    
    void run_all() {
        for (auto& k : registered_kernels_) {
            k();
        }
    }
private:
    std::vector<std::function<void()>> registered_kernels_;
};

class queue {
public:
    queue(device dev) : dev_(dev) {}
    device get_device() const { return dev_; }
    
    template <typename Func>
    void submit(Func f) {
        handler h;
        f(h);
        h.run_all();
    }
private:
    device dev_;
};

template <typename T>
class buffer {
public:
    buffer(T* data, size_t count) : data_(data), count_(count) {}
    size_t get_count() const { return count_; }
private:
    T* data_;
    size_t count_;
};

enum class access_mode {
    read,
    write,
    read_write
};

template <typename T, access_mode Mode>
class accessor {
public:
    accessor(buffer<T>& buf, handler& h) : count_(buf.get_count()) {}
    size_t size() const { return count_; }
private:
    size_t count_;
};

} // namespace sycl
```

**Step 4: Run test to verify it passes**
Run: `cmake --build build && ./build/test_runner`
Expected: Prints `test_sycl_mock PASS`

**Step 5: Commit**
```bash
git add include/panther_lake/sycl_mock.hpp tests/test_runner.cpp CMakeLists.txt
git commit -m "feat: implement SYCL programming model mockup"
```

---

### Task 3: Core CPU Architectures (Cougar Cove & Darkmont)

**Files:**
- Create: `include/panther_lake/cores.hpp`
- Create: `src/cores.cpp`
- Modify: `tests/test_runner.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`, append:
```cpp
#include "panther_lake/cores.hpp"

void test_cpu_cores() {
    panther_lake::CougarCoveCore p_core(0);
    panther_lake::DarkmontCore e_core(1, false);
    
    unsigned long long p_cycles = p_core.execute(10000, 2000);
    unsigned long long e_cycles = e_core.execute(10000, 2000);
    
    assert(p_cycles > 0);
    assert(e_cycles > 0);
    assert(p_core.get_power() > 0.0);
    assert(e_core.get_power() > 0.0);
    std::cout << "test_cpu_cores PASS" << std::endl;
}
```

Call `test_cpu_cores()` in `main()`.

**Step 2: Run test to verify it fails**
Run: `cmake --build build`
Expected: Compilation failure because `cores.hpp` does not exist.

**Step 3: Write minimal implementation**
`include/panther_lake/cores.hpp`:
```cpp
#pragma once
#include <string>

namespace panther_lake {

class CpuCore {
public:
    CpuCore(int core_id, std::string type, double frequency_ghz, double ipc_target);
    virtual ~CpuCore() = default;
    
    virtual unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) = 0;
    double get_power() const;
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

class CougarCoveCore : public CpuCore {
public:
    CougarCoveCore(int core_id, double frequency_ghz = 5.0);
    unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) override;
};

class DarkmontCore : public CpuCore {
public:
    DarkmontCore(int core_id, bool is_lp, double frequency_ghz = 3.5);
    unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) override;
private:
    bool is_lp_;
};

} // namespace panther_lake
```

`src/cores.cpp`:
```cpp
#include "panther_lake/cores.hpp"
#include <cmath>

namespace panther_lake {

CpuCore::CpuCore(int core_id, std::string type, double frequency_ghz, double ipc_target)
    : core_id_(core_id), type_(type), frequency_ghz_(frequency_ghz), ipc_target_(ipc_target) {}

double CpuCore::get_power() const {
    return active_power_watts_ + leakage_power_watts_;
}

CougarCoveCore::CougarCoveCore(int core_id, double frequency_ghz)
    : CpuCore(core_id, "P-Core (Cougar Cove)", frequency_ghz, 3.0) {
    leakage_power_watts_ = 0.5;
}

unsigned long long CougarCoveCore::execute(unsigned long long instructions, unsigned long long mem_ops) {
    unsigned long long hits = static_cast<unsigned long long>(mem_ops * 0.95);
    unsigned long long misses = mem_ops - hits;
    l1_hits_ += hits;
    l1_misses_ += misses;
    
    double exec_cycles = instructions / ipc_target_;
    double mem_penalty = (hits * 4) + (misses * 15);
    unsigned long long total_cycles = static_cast<unsigned long long>(exec_cycles + (mem_penalty * 0.4));
    
    cycles_elapsed_ += total_cycles;
    instructions_retired_ += instructions;
    active_power_watts_ = 4.0 * std::pow(frequency_ghz_ / 5.0, 3);
    
    return total_cycles;
}

DarkmontCore::DarkmontCore(int core_id, bool is_lp, double frequency_ghz)
    : CpuCore(core_id, is_lp ? "LP E-Core" : "E-Core (Darkmont)", is_lp ? 2.5 : frequency_ghz, is_lp ? 1.1 : 1.5), is_lp_(is_lp) {
    leakage_power_watts_ = is_lp ? 0.05 : 0.1;
}

unsigned long long DarkmontCore::execute(unsigned long long instructions, unsigned long long mem_ops) {
    unsigned long long hits = static_cast<unsigned long long>(mem_ops * 0.9);
    unsigned long long misses = mem_ops - hits;
    l1_hits_ += hits;
    l1_misses_ += misses;
    
    double exec_cycles = instructions / ipc_target_;
    double mem_penalty = (hits * 4) + (misses * 20);
    unsigned long long total_cycles = static_cast<unsigned long long>(exec_cycles + (mem_penalty * 0.8));
    
    cycles_elapsed_ += total_cycles;
    instructions_retired_ += instructions;
    
    double base_power = is_lp_ ? 0.5 : 1.2;
    active_power_watts_ = base_power * std::pow(frequency_ghz_ / 3.0, 3);
    
    return total_cycles;
}

} // namespace panther_lake
```

Link source directory in CMakeLists.txt:
```cmake
add_library(panther_lake_lib src/cores.cpp)
target_link_libraries(test_runner panther_lake_lib)
target_link_libraries(panther_lake_sim panther_lake_lib)
```

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build && ./build/test_runner`
Expected: Prints `test_cpu_cores PASS`

**Step 5: Commit**
```bash
git add include/panther_lake/cores.hpp src/cores.cpp tests/test_runner.cpp CMakeLists.txt
git commit -m "feat: add Cougar Cove and Darkmont C++ core simulators"
```

---

### Task 4: Xe3-LPG Graphics Tile Model

**Files:**
- Create: `include/panther_lake/gpu.hpp`
- Create: `src/gpu.cpp`
- Modify: `tests/test_runner.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`:
```cpp
#include "panther_lake/gpu.hpp"

void test_gpu() {
    panther_lake::Xe3GPU gpu(12, 2.5);
    unsigned long long cycles = gpu.execute(1000000, 50000);
    assert(cycles > 0);
    assert(gpu.get_power() > 0.0);
    assert(gpu.get_gflops() > 0.0);
    std::cout << "test_gpu PASS" << std::endl;
}
```

Call `test_gpu()` in `main()`.

**Step 2: Run test to verify it fails**
Run: `cmake --build build`
Expected: Compile error.

**Step 3: Write minimal implementation**
`include/panther_lake/gpu.hpp`:
```cpp
#pragma once

namespace panther_lake {

class Xe3GPU {
public:
    Xe3GPU(int xe_cores = 12, double frequency_ghz = 2.5);
    unsigned long long execute(unsigned long long ops, unsigned long long matrix_ops);
    double get_power() const;
    double get_gflops() const;
    int get_xe_cores() const { return xe_cores_; }

private:
    int xe_cores_;
    double frequency_ghz_;
    unsigned long long ops_completed_ = 0;
    unsigned long long cycles_elapsed_ = 0;
    double active_power_watts_ = 0.0;
    double leakage_power_watts_ = 0.0;
    
    int vector_engines_;
    int matrix_engines_;
    int ve_flops_per_cycle_ = 16;
    int xmx_flops_per_cycle_ = 128;
};

} // namespace panther_lake
```

`src/gpu.cpp`:
```cpp
#include "panther_lake/gpu.hpp"
#include <cmath>
#include <algorithm>

namespace panther_lake {

Xe3GPU::Xe3GPU(int xe_cores, double frequency_ghz)
    : xe_cores_(xe_cores), frequency_ghz_(frequency_ghz) {
    leakage_power_watts_ = 0.25 * xe_cores_;
    vector_engines_ = xe_cores_ * 8;
    matrix_engines_ = xe_cores_ * 8;
}

unsigned long long Xe3GPU::execute(unsigned long long ops, unsigned long long matrix_ops) {
    double vector_capacity = vector_engines_ * ve_flops_per_cycle_;
    double matrix_capacity = matrix_engines_ * xmx_flops_per_cycle_;
    
    double ve_cycles = vector_capacity > 0 ? ops / vector_capacity : 0;
    double xmx_cycles = matrix_capacity > 0 ? matrix_ops / matrix_capacity : 0;
    
    unsigned long long total_cycles = static_cast<unsigned long long>(
        std::max(ve_cycles, xmx_cycles) + 0.8 * std::min(ve_cycles, xmx_cycles)
    );
    
    if (total_cycles == 0 && (ops > 0 || matrix_ops > 0)) {
        total_cycles = 1;
    }
    
    cycles_elapsed_ += total_cycles;
    ops_completed_ += (ops + matrix_ops);
    
    double dynamic_power_factor = 2.5 * (static_cast<double>(xe_cores_) / 12.0);
    active_power_watts_ = dynamic_power_factor * std::pow(frequency_ghz_ / 2.5, 3);
    
    return total_cycles;
}

double Xe3GPU::get_power() const {
    return active_power_watts_ + leakage_power_watts_;
}

double Xe3GPU::get_gflops() const {
    double ve_gflops = vector_engines_ * ve_flops_per_cycle_ * frequency_ghz_;
    double xmx_gflops = matrix_engines_ * xmx_flops_per_cycle_ * frequency_ghz_;
    return ve_gflops + xmx_gflops;
}

} // namespace panther_lake
```

Modify `CMakeLists.txt` to add `src/gpu.cpp` to the library target:
```cmake
add_library(panther_lake_lib src/cores.cpp src/gpu.cpp)
```

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build && ./build/test_runner`
Expected: Prints `test_gpu PASS`

**Step 5: Commit**
```bash
git add include/panther_lake/gpu.hpp src/gpu.cpp tests/test_runner.cpp CMakeLists.txt
git commit -m "feat: implement Xe3-LPG graphics C++ model"
```

---

### Task 5: NPU 5 Execution Model

**Files:**
- Create: `include/panther_lake/npu.hpp`
- Create: `src/npu.cpp`
- Modify: `tests/test_runner.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`:
```cpp
#include "panther_lake/npu.hpp"

void test_npu() {
    panther_lake::NPU5 npu(1.8);
    unsigned long long cycles = npu.execute_matmul(128, 128, 128);
    assert(cycles > 0);
    assert(npu.get_power() > 0.0);
    assert(npu.get_max_tops() >= 40.0);
    std::cout << "test_npu PASS" << std::endl;
}
```

Call `test_npu()` in `main()`.

**Step 2: Run test to verify it fails**
Expected: Compile error.

**Step 3: Write minimal implementation**
`include/panther_lake/npu.hpp`:
```cpp
#pragma once

namespace panther_lake {

class NPU5 {
public:
    NPU5(double frequency_ghz = 1.8);
    unsigned long long execute_matmul(int m, int n, int k);
    double get_power() const;
    double get_max_tops() const;
    unsigned long long get_macs_completed() const { return macs_completed_; }

private:
    double frequency_ghz_;
    unsigned long long macs_completed_ = 0;
    unsigned long long cycles_elapsed_ = 0;
    double active_power_watts_ = 0.0;
    double leakage_power_watts_ = 0.2;
    
    int num_nce_ = 4;
    int macs_per_nce_cycle_ = 4096;
};

} // namespace panther_lake
```

`src/npu.cpp`:
```cpp
#include "panther_lake/npu.hpp"
#include <algorithm>

namespace panther_lake {

NPU5::NPU5(double frequency_ghz) : frequency_ghz_(frequency_ghz) {}

unsigned long long NPU5::execute_matmul(int m, int n, int k) {
    unsigned long long total_macs = static_cast<unsigned long long>(m) * n * k;
    double capacity_per_cycle = num_nce_ * macs_per_nce_cycle_;
    
    double compute_cycles = total_macs / capacity_per_cycle;
    unsigned long long total_cycles = static_cast<unsigned long long>(compute_cycles * 1.1);
    
    if (total_cycles == 0 && total_macs > 0) {
        total_cycles = 1;
    }
    
    cycles_elapsed_ += total_cycles;
    macs_completed_ += total_macs;
    
    double utilization = std::min(1.0, compute_cycles / std::max(1ULL, total_cycles));
    active_power_watts_ = 8.0 * (frequency_ghz_ / 1.8) * utilization;
    
    return total_cycles;
}

double NPU5::get_power() const {
    return active_power_watts_ + leakage_power_watts_;
}

double NPU5::get_max_tops() const {
    double ops_per_cycle = num_nce_ * macs_per_nce_cycle_ * 2;
    return (ops_per_cycle * frequency_ghz_) / 1000.0;
}

} // namespace panther_lake
```

Modify `CMakeLists.txt` library configuration:
```cmake
add_library(panther_lake_lib src/cores.cpp src/gpu.cpp src/npu.cpp)
```

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build && ./build/test_runner`
Expected: Prints `test_npu PASS`

**Step 5: Commit**
```bash
git add include/panther_lake/npu.hpp src/npu.cpp tests/test_runner.cpp CMakeLists.txt
git commit -m "feat: implement NPU 5 accelerator C++ model"
```

---

### Task 6: Memory Hierarchy & SLC

**Files:**
- Create: `include/panther_lake/memory.hpp`
- Create: `src/memory.cpp`
- Modify: `tests/test_runner.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`:
```cpp
#include "panther_lake/memory.hpp"

void test_memory() {
    panther_lake::LPDDR5XController memory(9600.0, 2);
    assert(memory.get_peak_bandwidth_gbs() > 150.0);
    
    panther_lake::SystemLevelCache slc(24);
    auto [hit, cycles] = slc.access(0x1000);
    assert(cycles > 0);
    std::cout << "test_memory PASS" << std::endl;
}
```

Call `test_memory()` in `main()`.

**Step 2: Run test to verify it fails**
Compile should fail.

**Step 3: Write minimal implementation**
`include/panther_lake/memory.hpp`:
```cpp
#pragma once
#include <cstddef>
#include <unordered_set>
#include <utility>

namespace panther_lake {

class LPDDR5XController {
public:
    LPDDR5XController(double speed_mts = 9600.0, int channels = 2);
    double get_peak_bandwidth_gbs() const;
    double access_latency_ns(bool is_write) const;
    void transfer(size_t size_bytes, bool is_write);
    
    unsigned long long get_reads() const { return reads_; }
    unsigned long long get_writes() const { return writes_; }
    unsigned long long get_bytes_transferred() const { return bytes_transferred_; }

private:
    double speed_mts_;
    int channels_;
    int data_width_bits_ = 64;
    unsigned long long reads_ = 0;
    unsigned long long writes_ = 0;
    unsigned long long bytes_transferred_ = 0;
};

class SystemLevelCache {
public:
    SystemLevelCache(int size_mb = 24);
    std::pair<bool, int> access(unsigned long long address);
    
    unsigned long long get_hits() const { return hits_; }
    unsigned long long get_misses() const { return misses_; }
    int get_size_mb() const { return size_mb_; }

private:
    int size_mb_;
    int latency_cycles_ = 45;
    unsigned long long hits_ = 0;
    unsigned long long misses_ = 0;
    std::unordered_set<unsigned long long> cached_lines_;
};

} // namespace panther_lake
```

`src/memory.cpp`:
```cpp
#include "panther_lake/memory.hpp"
#include <cstdlib>

namespace panther_lake {

LPDDR5XController::LPDDR5XController(double speed_mts, int channels)
    : speed_mts_(speed_mts), channels_(channels) {}

double LPDDR5XController::get_peak_bandwidth_gbs() const {
    double width_bytes = data_width_bits_ / 8.0;
    return (speed_mts_ * 1000000.0 * width_bytes * channels_) / 1000000000.0;
}

double LPDDR5XController::access_latency_ns(bool is_write) const {
    double base = is_write ? 35.0 : 45.0;
    double queue_delay = 5.0 + (std::rand() % 10);
    return base + queue_delay;
}

void LPDDR5XController::transfer(size_t size_bytes, bool is_write) {
    if (is_write) {
        writes_++;
    } else {
        reads_++;
    }
    bytes_transferred_ += size_bytes;
}

SystemLevelCache::SystemLevelCache(int size_mb) : size_mb_(size_mb) {}

std::pair<bool, int> SystemLevelCache::access(unsigned long long address) {
    unsigned long long line = address >> 6;
    if (cached_lines_.count(line)) {
        hits_++;
        return {true, latency_cycles_};
    } else {
        misses_++;
        if (cached_lines_.size() > (size_mb_ * 1024ULL * 1024ULL / 64ULL)) {
            // Simple eviction
            cached_lines_.erase(cached_lines_.begin());
        }
        cached_lines_.insert(line);
        return {false, latency_cycles_};
    }
}

} // namespace panther_lake
```

Modify `CMakeLists.txt`:
```cmake
add_library(panther_lake_lib src/cores.cpp src/gpu.cpp src/npu.cpp src/memory.cpp)
```

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build && ./build/test_runner`
Expected: Prints `test_memory PASS`

**Step 5: Commit**
```bash
git add include/panther_lake/memory.hpp src/memory.cpp tests/test_runner.cpp CMakeLists.txt
git commit -m "feat: implement C++ LPDDR5X and SLC cache models"
```

---

### Task 7: Foveros 3D Interconnect

**Files:**
- Create: `include/panther_lake/fabric.hpp`
- Create: `src/fabric.cpp`
- Modify: `tests/test_runner.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`:
```cpp
#include "panther_lake/fabric.hpp"

void test_fabric() {
    panther_lake::FoverosInterconnect fabric;
    int lat = fabric.route("ComputeTile", "GraphicsTile", 64);
    assert(lat > 0);
    assert(fabric.get_total_bytes_routed() == 64);
    std::cout << "test_fabric PASS" << std::endl;
}
```

Call `test_fabric()` in `main()`.

**Step 2: Run test to verify it fails**
Compile should fail.

**Step 3: Write minimal implementation**
`include/panther_lake/fabric.hpp`:
```cpp
#pragma once
#include <string>

namespace panther_lake {

class FoverosInterconnect {
public:
    FoverosInterconnect();
    int route(const std::string& src, const std::string& dst, unsigned long long data_size_bytes);
    
    unsigned long long get_transfers_count() const { return transfers_count_; }
    unsigned long long get_total_bytes_routed() const { return total_bytes_routed_; }
    unsigned long long get_contention_count() const { return contention_count_; }

private:
    unsigned long long transfers_count_ = 0;
    unsigned long long total_bytes_routed_ = 0;
    unsigned long long contention_count_ = 0;
};

} // namespace panther_lake
```

`src/fabric.cpp`:
```cpp
#include "panther_lake/fabric.hpp"
#include <cmath>

namespace panther_lake {

FoverosInterconnect::FoverosInterconnect() {}

int FoverosInterconnect::route(const std::string& src, const std::string& dst, unsigned long long data_size_bytes) {
    transfers_count_++;
    total_bytes_routed_ += data_size_bytes;
    
    double bandwidth = 128.0; // GB/s base
    int base_lat = 12;
    
    if ((src == "ComputeTile" && dst == "GraphicsTile") || (src == "GraphicsTile" && dst == "ComputeTile")) {
        bandwidth = 256.0;
        base_lat = 8;
    } else if ((src == "GraphicsTile" && dst == "PlatformControllerTile") || (src == "PlatformControllerTile" && dst == "GraphicsTile")) {
        bandwidth = 64.0;
        base_lat = 16;
    }
    
    double transfer_ns = (data_size_bytes / (bandwidth * 1000000000.0)) * 1000000000.0;
    int transfer_cycles = static_cast<int>(transfer_ns / 0.5); // 2.0 GHz base fabric clock
    
    int contention_cycles = 0;
    if (transfers_count_ % 100 == 0) {
        contention_count_++;
        contention_cycles = 5;
    }
    
    return base_lat + transfer_cycles + contention_cycles;
}

} // namespace panther_lake
```

Modify `CMakeLists.txt`:
```cmake
add_library(panther_lake_lib src/cores.cpp src/gpu.cpp src/npu.cpp src/memory.cpp src/fabric.cpp)
```

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build && ./build/test_runner`
Expected: Prints `test_fabric PASS`

**Step 5: Commit**
```bash
git add include/panther_lake/fabric.hpp src/fabric.cpp tests/test_runner.cpp CMakeLists.txt
git commit -m "feat: implement Foveros 3D Interconnect C++ model"
```

---

### Task 8: Platform Controller Tile

**Files:**
- Create: `include/panther_lake/platform.hpp`
- Create: `src/platform.cpp`
- Modify: `tests/test_runner.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`:
```cpp
#include "panther_lake/platform.hpp"

void test_platform() {
    panther_lake::PlatformControllerTile pct;
    int cycles = pct.simulate_io(1000000, 500000);
    assert(cycles > 0);
    assert(pct.get_power() > 0.0);
    assert(pct.calculate_dvfs_voltage(5.0) > 0.7);
    std::cout << "test_platform PASS" << std::endl;
}
```

Call `test_platform()` in `main()`.

**Step 2: Run test to verify it fails**
Compile should fail.

**Step 3: Write minimal implementation**
`include/panther_lake/platform.hpp`:
```cpp
#pragma once
#include <string>

namespace panther_lake {

class PlatformControllerTile {
public:
    PlatformControllerTile();
    int simulate_io(unsigned long long pcie_traffic_bytes, unsigned long long tb_traffic_bytes);
    double calculate_dvfs_voltage(double freq_ghz) const;
    double get_power() const;

private:
    unsigned long long pcie_traffic_ = 0;
    unsigned long long tb_traffic_ = 0;
    double active_power_watts_ = 0.5;
    double leakage_power_watts_ = 0.3;
    
    double pcie_max_gbs_ = 63.0;
    double tb_max_gbs_ = 10.0;
};

} // namespace panther_lake
```

`src/platform.cpp`:
```cpp
#include "panther_lake/platform.hpp"
#include <algorithm>

namespace panther_lake {

PlatformControllerTile::PlatformControllerTile() {}

int PlatformControllerTile::simulate_io(unsigned long long pcie_traffic_bytes, unsigned long long tb_traffic_bytes) {
    pcie_traffic_ += pcie_traffic_bytes;
    tb_traffic_ += tb_traffic_bytes;
    
    double pcie_transfer_ns = (pcie_traffic_bytes / (pcie_max_gbs_ * 1000000000.0)) * 1000000000.0;
    double tb_transfer_ns = (tb_traffic_bytes / (tb_max_gbs_ * 1000000000.0)) * 1000000000.0;
    
    int pct_cycles = static_cast<int>(std::max(pcie_transfer_ns, tb_transfer_ns));
    
    double total_traffic = pcie_traffic_bytes + tb_traffic_bytes;
    double io_utilization = std::min(1.0, total_traffic / 10000000.0);
    active_power_watts_ = 0.5 + 2.0 * io_utilization;
    
    return std::max(1, pct_cycles);
}

double PlatformControllerTile::calculate_dvfs_voltage(double freq_ghz) const {
    if (freq_ghz <= 1.0) return 0.70;
    if (freq_ghz >= 6.0) return 1.35;
    double ratio = (freq_ghz - 1.0) / 5.0;
    return 0.70 + ratio * (1.35 - 0.70);
}

double PlatformControllerTile::get_power() const {
    return active_power_watts_ + leakage_power_watts_;
}

} // namespace panther_lake
```

Modify `CMakeLists.txt`:
```cmake
add_library(panther_lake_lib src/cores.cpp src/gpu.cpp src/npu.cpp src/memory.cpp src/fabric.cpp src/platform.cpp)
```

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build && ./build/test_runner`
Expected: Prints `test_platform PASS`

**Step 5: Commit**
```bash
git add include/panther_lake/platform.hpp src/platform.cpp tests/test_runner.cpp CMakeLists.txt
git commit -m "feat: implement Platform Controller Tile C++ model"
```

---

### Task 9: Simulator Orchestrator

**Files:**
- Create: `include/panther_lake/simulator.hpp`
- Create: `src/simulator.cpp`
- Modify: `tests/test_runner.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**
In `tests/test_runner.cpp`:
```cpp
#include "panther_lake/simulator.hpp"

void test_simulator() {
    panther_lake::PantherLakeProcessor processor(true);
    panther_lake::Workload workload;
    workload.cpu_instructions = 500000;
    workload.cpu_mem_ops = 100000;
    workload.gpu_ops = 100000;
    workload.gpu_matrix_ops = 20000;
    workload.npu_m = 128; workload.npu_n = 128; workload.npu_k = 128;
    workload.pcie_bytes = 50000;
    workload.tb_bytes = 10000;
    
    auto res = processor.run_workload(workload);
    assert(res.total_cycles > 0);
    assert(res.avg_power_watts > 0.0);
    assert(res.total_energy_joules > 0.0);
    std::cout << "test_simulator PASS" << std::endl;
}
```

Call `test_simulator()` in `main()`.

**Step 2: Run test to verify it fails**
Compile should fail.

**Step 3: Write minimal implementation**
`include/panther_lake/simulator.hpp`:
```cpp
#pragma once
#include <vector>
#include <memory>
#include "panther_lake/cores.hpp"
#include "panther_lake/gpu.hpp"
#include "panther_lake/npu.hpp"
#include "panther_lake/memory.hpp"
#include "panther_lake/fabric.hpp"
#include "panther_lake/platform.hpp"

namespace panther_lake {

struct Workload {
    unsigned long long cpu_instructions = 0;
    unsigned long long cpu_mem_ops = 0;
    unsigned long long gpu_ops = 0;
    unsigned long long gpu_matrix_ops = 0;
    int npu_m = 0;
    int npu_n = 0;
    int npu_k = 0;
    unsigned long long pcie_bytes = 0;
    unsigned long long tb_bytes = 0;
};

struct SimulationResult {
    unsigned long long total_cycles;
    double simulation_time_sec;
    double avg_power_watts;
    double total_energy_joules;
    double cpu_ips;
    double gpu_gflops;
    double npu_tops;
    double peak_temp_c;
};

class PantherLakeProcessor {
public:
    PantherLakeProcessor(bool high_power_config = true);
    SimulationResult run_workload(const Workload& workload);

    const std::vector<std::unique_ptr<CougarCoveCore>>& get_p_cores() const { return p_cores_; }
    const std::vector<std::unique_ptr<DarkmontCore>>& get_e_cores() const { return e_cores_; }
    const std::vector<std::unique_ptr<DarkmontCore>>& get_lp_cores() const { return lp_cores_; }
    const Xe3GPU& get_gpu() const { return gpu_; }
    const NPU5& get_npu() const { return npu_; }
    const LPDDR5XController& get_memory_ctrl() const { return memory_ctrl_; }
    const SystemLevelCache& get_slc() const { return slc_; }
    const FoverosInterconnect& get_fabric() const { return fabric_; }
    const PlatformControllerTile& get_pct() const { return pct_; }

private:
    bool high_power_config_;
    std::vector<std::unique_ptr<CougarCoveCore>> p_cores_;
    std::vector<std::unique_ptr<DarkmontCore>> e_cores_;
    std::vector<std::unique_ptr<DarkmontCore>> lp_cores_;
    Xe3GPU gpu_;
    NPU5 npu_;
    LPDDR5XController memory_ctrl_;
    SystemLevelCache slc_;
    FoverosInterconnect fabric_;
    PlatformControllerTile pct_;
};

} // namespace panther_lake
```

`src/simulator.cpp`:
```cpp
#include "panther_lake/simulator.hpp"
#include <algorithm>
#include <cstdlib>

namespace panther_lake {

PantherLakeProcessor::PantherLakeProcessor(bool high_power_config)
    : high_power_config_(high_power_config),
      gpu_(high_power_config ? 12 : 4, high_power_config ? 2.5 : 2.0),
      npu_(1.8),
      memory_ctrl_(9600.0, 2),
      slc_(high_power_config ? 24 : 16) {
      
    for (int i = 0; i < 4; ++i) {
        p_cores_.push_back(std::make_unique<CougarCoveCore>(i, 5.0));
    }
    
    if (high_power_config_) {
        for (int i = 0; i < 8; ++i) {
            e_cores_.push_back(std::make_unique<DarkmontCore>(i, false, 3.5));
        }
        for (int i = 0; i < 4; ++i) {
            lp_cores_.push_back(std::make_unique<DarkmontCore>(i, true, 2.5));
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            lp_cores_.push_back(std::make_unique<DarkmontCore>(i, true, 2.5));
        }
    }
}

SimulationResult PantherLakeProcessor::run_workload(const Workload& workload) {
    unsigned long long cpu_cycles = 0;
    
    // CPU work sharing
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
        
        unsigned long long lp_work = static_cast<unsigned long long>(workload.cpu_instructions * 0.05 / lp_cores_.size());
        unsigned long long lp_mem = static_cast<unsigned long long>(workload.cpu_mem_ops * 0.05 / lp_cores_.size());
        for (auto& core : lp_cores_) {
            cpu_cycles = std::max(cpu_cycles, core->execute(lp_work, lp_mem));
        }
    } else {
        unsigned long long lp_work = static_cast<unsigned long long>(workload.cpu_instructions * 0.3 / lp_cores_.size());
        unsigned long long lp_mem = static_cast<unsigned long long>(workload.cpu_mem_ops * 0.3 / lp_cores_.size());
        for (auto& core : lp_cores_) {
            cpu_cycles = std::max(cpu_cycles, core->execute(lp_work, lp_mem));
        }
    }
    
    unsigned long long gpu_cycles = gpu_.execute(workload.gpu_ops, workload.gpu_matrix_ops);
    unsigned long long npu_cycles = npu_.execute_matmul(workload.npu_m, workload.npu_n, workload.npu_k);
    unsigned long long pct_cycles = pct_.simulate_io(workload.pcie_bytes, workload.tb_bytes);
    
    // Cache miss routing to SLC
    unsigned long long cpu_misses = 0;
    for (auto& core : p_cores_) cpu_misses += core->get_l1_misses();
    for (auto& core : e_cores_) cpu_misses += core->get_l1_misses();
    for (auto& core : lp_cores_) cpu_misses += core->get_l1_misses();
    
    unsigned long long slc_access_cycles = 0;
    for (unsigned long long i = 0; i < cpu_misses; ++i) {
        unsigned long long fake_addr = std::rand();
        auto [hit, lat] = slc_.access(fake_addr);
        slc_access_cycles += lat;
        if (!hit) {
            memory_ctrl_.transfer(64, false);
        }
    }
    
    int fabric_cycles = fabric_.route("ComputeTile", "GraphicsTile", workload.gpu_ops * 4) +
                        fabric_.route("ComputeTile", "PlatformControllerTile", workload.pcie_bytes + workload.tb_bytes);
                        
    unsigned long long total_cycles = std::max({cpu_cycles, gpu_cycles, npu_cycles, pct_cycles}) + fabric_cycles + slc_access_cycles;
    
    double total_power = 0.0;
    for (auto& core : p_cores_) total_power += core->get_power();
    for (auto& core : e_cores_) total_power += core->get_power();
    for (auto& core : lp_cores_) total_power += core->get_power();
    total_power += gpu_.get_power() + npu_.get_power() + pct_.get_power();
    
    double simulation_time_sec = total_cycles / (2.0 * 1000000000.0);
    double total_energy_joules = total_power * simulation_time_sec;
    
    double cpu_ips = workload.cpu_instructions / simulation_time_sec;
    double gpu_gflops = (workload.gpu_ops + workload.gpu_matrix_ops * 2) / (simulation_time_sec * 1000000000.0);
    double npu_tops = (static_cast<double>(workload.npu_m) * workload.npu_n * workload.npu_k * 2) / (simulation_time_sec * 1000000000000.0);
    
    double peak_temp_c = 35.0 + 1.2 * total_power;
    
    return SimulationResult{
        total_cycles,
        simulation_time_sec,
        total_power,
        total_energy_joules,
        cpu_ips,
        gpu_gflops,
        npu_tops,
        peak_temp_c
    };
}

} // namespace panther_lake
```

Modify `CMakeLists.txt`:
```cmake
add_library(panther_lake_lib src/cores.cpp src/gpu.cpp src/npu.cpp src/memory.cpp src/fabric.cpp src/platform.cpp src/simulator.cpp)
```

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build && ./build/test_runner`
Expected: Prints `test_simulator PASS`

**Step 5: Commit**
```bash
git add include/panther_lake/simulator.hpp src/simulator.cpp tests/test_runner.cpp CMakeLists.txt
git commit -m "feat: implement C++ processor simulator coordinator"
```

---

### Task 10: CLI Application & Workloads

**Files:**
- Modify: `main.cpp`
- Create: `DESIGN.md`
- Create: `README.md`

**Step 1: Write the failing test**
N/A

**Step 2: Run test to verify it fails**
N/A

**Step 3: Write minimal implementation**
Write standard main function executing sample oneAPI CPU/GPU/NPU kernel structures on the simulator.

`main.cpp`:
```cpp
#include <iostream>
#include <iomanip>
#include "panther_lake/sycl_mock.hpp"
#include "panther_lake/simulator.hpp"

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "   Intel Panther Lake - oneAPI C++ Architectural Simulator" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    panther_lake::PantherLakeProcessor processor(true);
    
    // Simulate a oneAPI SYCL-like program queue submission
    sycl::device dev(sycl::device_type::gpu);
    sycl::queue q(dev);
    
    std::cout << "Dispatching oneAPI program to: " << q.get_device().get_name() << std::endl;
    
    // Build a mock workload representing the kernel dispatch
    panther_lake::Workload workload;
    workload.cpu_instructions = 100000000ULL;
    workload.cpu_mem_ops = 25000000ULL;
    workload.gpu_ops = 800000000ULL;
    workload.gpu_matrix_ops = 200000000ULL;
    workload.npu_m = 2048;
    workload.npu_n = 2048;
    workload.npu_k = 2048;
    workload.pcie_bytes = 50000000ULL;
    workload.tb_bytes = 10000000ULL;
    
    q.submit([&workload](sycl::handler& h) {
        h.parallel_for<class AIKernel>(workload.gpu_ops, [](size_t idx) {
            // Mock GPU kernel execution body
        });
    });
    
    std::cout << "Simulating hardware execution on Panther Lake processor..." << std::endl;
    auto res = processor.run_workload(workload);
    
    std::cout << "\n================ SIMULATION STATISTICS ================" << std::endl;
    std::cout << "Cycles elapsed:       " << res.total_cycles << " cycles" << std::endl;
    std::cout << "Execution time:       " << res.simulation_time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "Average Power:        " << res.avg_power_watts << " W" << std::endl;
    std::cout << "Total Energy:         " << res.total_energy_joules * 1000.0 << " mJ" << std::endl;
    std::cout << "CPU Throughput:       " << res.cpu_ips / 1000000.0 << " MIPS" << std::endl;
    std::cout << "GPU Performance:      " << res.gpu_gflops << " GFLOPS" << std::endl;
    std::cout << "NPU Performance:      " << res.npu_tops << " TOPS" << std::endl;
    std::cout << "Estimated Temperature: " << res.peak_temp_c << " C" << std::endl;
    std::cout << "=======================================================\n" << std::endl;
    
    return 0;
}
```

Create `DESIGN.md` explaining oneAPI integration, core details, packaging links.
Create `README.md` with instructions on compiling and executing.

**Step 4: Run test to verify it passes**
Run: `cmake -B build; cmake --build build && ./build/panther_lake_sim`
Expected: Compiles and runs correctly, outputting stats.

**Step 5: Commit**
```bash
git add main.cpp DESIGN.md README.md
git commit -m "docs: finalize design documentation and simulator executable"
```
