# Intel-Only Xe3 GPU Core Refactor Implementation Plan

**Goal:** Refactor the custom Hybrid AI Superchip simulator to utilize an Intel-only Xe3 (Celestial) GPU core model instead of the Nvidia Blackwell GPU core model, making the SoC package fully Intel-focused.

**Architecture:**
- Rename `BlackwellGpuCore` to `Xe3GpuCore` representing Intel's Xe3 (Celestial) architecture.
- Adjust the GPU core specifications to match Intel Xe3: 8 Vector Engines (128 FLOPs/cycle total) and 8 Matrix Engines (1024 FLOPs/cycle total) per core.
- Update the power profile of the GPU core to fit an integrated Intel GPU design (1.8W dynamic power at 1.6 GHz, scaling with frequency cubed).
- Update the terminal telemetry dashboard and design documentation to reflect the Intel-only design.

---

## Action Items

- [ ] **Step 1: Refactor cores header**
  - Modify `include/panther_lake/cores.hpp` to rename `BlackwellGpuCore` to `Xe3GpuCore`.
  - Update its vector/matrix FLOPs calculations based on Xe3 Celestial architecture specifications.

- [ ] **Step 2: Refactor cores implementation**
  - Modify `src/cores.cpp` to rename and implement `Xe3GpuCore` functions.
  - Set the core type name to `"Intel Xe3 GPU Core"` and adjust power scaling parameters to represent an integrated Intel GPU.

- [ ] **Step 3: Refactor GPU tile manager**
  - Modify `include/panther_lake/gpu.hpp` and `src/gpu.cpp` to instantiate and manage `Xe3GpuCore` instances instead of Blackwell.
  - Update `Xe3GPU::execute` workload partitioning logic based on Intel Xe3 Vector/Matrix Engine throughputs.

- [ ] **Step 4: Update test runner**
  - Modify `tests/test_runner.cpp` to rename the tests (e.g., `test_xe3_gpu_core`) and verify the functionality of the new `Xe3GpuCore`.

- [ ] **Step 5: Update main dashboard telemetry**
  - Modify `main.cpp` to display `"Intel Xe3 GPU Cores"` and `"(Intel Xe3 Celestial Graphics)"` instead of Blackwell.

- [ ] **Step 6: Update design documentation**
  - Modify `DESIGN_SUPERCHIP.md` to reflect the transition from Blackwell to the Intel-only Xe3 GPU.

- [ ] **Step 7: Verification**
  - Build the project and run the tests to verify compilation and correctness.
  - Run the simulator binary to inspect the new console dashboard.
