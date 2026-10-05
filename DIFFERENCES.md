# Intel Panther Lake Simulator: Architectural Differences Report

This document records the architectural and software changes made to the Intel Panther Lake C++ oneAPI Simulator relative to the original Panther Lake design.

---

## 1. Unified Core Hierarchy & Polymorphism

*   **Original Design:** Only CPU cores inherited from a base `CpuCore` class, and the GPU was modeled monolithically inside a separate `Xe3GPU` class.
*   **Modified Design:** Refactored the core class hierarchy to introduce a polymorphic [`ExecutionCore`](include/panther_lake/cores.hpp) base class. A dedicated [`Xe3GpuCore`](include/panther_lake/cores.hpp) subclass was added. Both CPU cores ([`CougarCoveCore`](include/panther_lake/cores.hpp), [`DarkmontCore`](include/panther_lake/cores.hpp)) and GPU cores now inherit from [`ExecutionCore`](include/panther_lake/cores.hpp).
*   **Rationale:** Allows uniform frequency scaling, active power calculations, and core gating status tracking under a single interface.

---

## 2. Dynamic Voltage and Frequency Scaling (DVFS)

*   **Original Design:** Simulated components executed at fixed clock frequencies.
*   **Modified Design:** Implemented a **DVFS Power-Throttling Loop** inside [`run_workload()`](src/simulator.cpp). The loop continuously checks the total package power draw against a target TDP. If the target is exceeded, the simulator dynamically steps down core frequencies (down to a minimum floor of 1.5 GHz for CPU P-cores, 1.0 GHz for E-cores, and 0.8 GHz for GPU cores).
*   **Rationale:** Models realistic thermal and power constraints of a system running under fixed power envelopes.

---

## 3. Dynamic Memory Bandwidth Partitioning

*   **Original Design:** Fixed memory bandwidth allocation.
*   **Modified Design:** Implemented a **dynamic bandwidth partitioner** in [`run_workload()`](src/simulator.cpp) allocating the LPDDR5X-9600 controller's peak bandwidth (153.6 GB/s) based on the active workload phase:
    *   **NPU active only:** Shifts **85% of peak bandwidth** to the NPU.
    *   **GPU active only:** Shifts **85% of peak bandwidth** to the GPU.
    *   **GPU & NPU both active:** Splits the bandwidth (**60% to GPU, 30% to NPU**) to handle bus contention.
*   **Rationale:** Minimizes memory bottleneck stalls during highly asymmetric phases of workload execution (e.g. NPU-bound prompt processing vs. GPU-bound autoregressive token generation).

---

## 4. Battery-Saver Mode (15W TDP Limit)

*   **Original Design:** No energy-saving mode.
*   **Modified Design:** Added a **Battery-Saver Mode** configuration to the [`Workload`](include/panther_lake/simulator.hpp) struct. When active, it triggers the following architectural changes:
    *   **Core Gating:** Power-gates all high-power [`CougarCoveCore`](include/panther_lake/cores.hpp) P-cores (dynamic and leakage power clamped to `0.0W` using a `gated_` status flag).
    *   **Workload Re-routing:** Automatically routes CPU execution entirely to the [`DarkmontCore`](include/panther_lake/cores.hpp) E-cores (70%) and LP E-cores (30%).
    *   **Model Precision Scaling:** Simulates **INT4 weight quantization** (0.5 bytes per parameter) instead of FP16 (2.0 bytes per parameter) for the 40 Billion parameter LLM workload. 
    *   **Performance Impact:** Shrinking the model size from 80 GB to 20 GB reduces memory controller traffic by 4×. This removes the memory bandwidth bottleneck, resulting in **34% faster execution** (2,093.0 ms vs. 3,170.1 ms in 35W mode) despite heavily throttled core frequencies.

---

## 5. Active Power Initialization & Reset States

*   **Original Design:** Component power states persisted across workload simulator dispatches, potentially carrying over dynamic power calculations.
*   **Modified Design:** Added a [`reset_active_power()`](include/panther_lake/cores.hpp) method across all execution cores and tiles ([`NPU5`](include/panther_lake/npu.hpp) and [`PlatformControllerTile`](include/panther_lake/platform.hpp)). This method is called at the beginning of each simulation run to reset component power draws to nominal, workload-ready states, eliminating state leakage.

---

## 6. Project Files Map

Below is the file footprint of the architectural simulator:

| File | Purpose |
| :--- | :--- |
| [`include/panther_lake/cores.hpp`](include/panther_lake/cores.hpp) | Definition of polymorphic `ExecutionCore` classes and subclasses. |
| [`src/cores.cpp`](src/cores.cpp) | Cycle & power simulation implementation for CPU and GPU cores. |
| [`include/panther_lake/gpu.hpp`](include/panther_lake/gpu.hpp) | Definition of `Xe3GPU` tile manager managing a collection of `Xe3GpuCore` instances. |
| [`src/gpu.cpp`](src/gpu.cpp) | Directs vector/matrix operations across GPU cores. |
| [`src/simulator.cpp`](src/simulator.cpp) | Orchestrates dynamic bandwidth partitioning, DVFS throttling, and workload routing. |
| [`main.cpp`](main.cpp) | Comparative execution entry point testing Normal and Battery-Saver Modes. |
| [`tests/test_runner.cpp`](tests/test_runner.cpp) | Verification suite asserting TDP limits, power-gating correctness, and polymorphism. |
| [`docs/simulation_outcomes.md`](docs/simulation_outcomes.md) | Formatted report detailing simulated LLM execution metrics. |
