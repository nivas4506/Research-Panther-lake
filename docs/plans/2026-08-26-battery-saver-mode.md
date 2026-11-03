# Battery-Saver Mode Implementation Plan

**Goal:** Implement a high-efficiency Battery-Saver Mode in the Panther Lake simulator that applies a 15W TDP limit, power-gates the high-power Cougar Cove P-cores, and downscales memory access footprints via simulated INT4 quantization.

**Architecture:**
- Add `bool battery_saver = false;` to the `Workload` specification struct.
- Introduce gating state management (`gated_` member and `set_gated`/`is_gated` accessors) in the base `ExecutionCore` class.
- Modify `ExecutionCore::get_power()` to return `0.0W` if a core is gated.
- Refactor `PantherLakeProcessor::run_workload` to:
  - Clamp package power target at `15.0W` instead of `35.0W`.
  - Shut down P-cores completely when battery saver is active.
  - Route CPU workloads dynamically to E-cores and LP-cores instead of P-cores.
  - Scale down simulated GPU/NPU weights transfer footprint by 4× (from 2.0 bytes for FP16 to 0.5 bytes for INT4) to save memory controller power.
  - Restore core states and gating status back to default at the end of each run.

---

## Action Items

- [ ] **Step 1: Update cores header and implementation**
  - Edit `include/panther_lake/cores.hpp` to add `gated_` state and `set_gated`/`is_gated` accessors to the `ExecutionCore` base class.
  - Edit `src/cores.cpp` to check `gated_` status in `get_power()`.

- [ ] **Step 2: Update simulator header**
  - Edit `include/panther_lake/simulator.hpp` to add `bool battery_saver = false;` to the `Workload` struct.

- [ ] **Step 3: Update simulator run_workload implementation**
  - Edit `src/simulator.cpp` to scale memory transfer bytes down by 4× for INT4 when `workload.battery_saver` is active.
  - Edit `src/simulator.cpp` to set `target_tdp = 15.0` and gate P-cores if `workload.battery_saver` is active.
  - Update CPU execution routing to bypass gated P-cores.
  - Restore gated status for P-cores at the end of the simulation.

- [ ] **Step 4: Update test runner**
  - Add a dedicated test `test_battery_saver_mode()` in `tests/test_runner.cpp` to verify:
    - Power is capped under 15W.
    - P-cores consume 0W power.
    - CPU operations are successfully routed and execute.
  - Call the new test in `main()`.

- [ ] **Step 5: Add Battery-Saver Run to main application**
  - Update `main.cpp` to run a comparison run: first executing the standard 40B LLM workload in Normal Mode (35W), and then running the same workload in Battery-Saver Mode (15W ULP with P-core gating and INT4 precision).
  - Print telemetry side-by-side or sequentially for comparison.

- [ ] **Step 6: Verification**
  - Build the project and run all tests to confirm they pass.
  - Execute the simulator binary to inspect the new comparative telemetry outputs.
