# Intel Panther Lake oneAPI C++ Simulator

A modular, cycle-approximate C++17 architectural simulator representing Intel's Panther Lake processor (Core Ultra Series 3) disaggregated tile structures, modeling CPU cores (Cougar Cove & Darkmont), Xe3-LPG GPU, NPU 5, LPDDR5X, and Foveros 3D Interconnect. It supports simulating oneAPI (SYCL) compute queue dispatches.

## Directory Structure

- **`include/panther_lake/`**: Header declarations.
  - `sycl_mock.hpp`: Mocks SYCL namespace queues, devices, buffers, accessors.
  - `cores.hpp`: CPU cores (Cougar Cove & Darkmont).
  - `gpu.hpp`: Xe3-LPG Celestial graphics tile.
  - `npu.hpp`: NPU 5 accelerator.
  - `memory.hpp`: Memory controller (LPDDR5X-9600) and System Level Cache (SLC).
  - `fabric.hpp`: Foveros 3D Packaging Interconnect.
  - `platform.hpp`: Platform Controller Tile (PCT) & DVFS scaling.
  - `simulator.hpp`: Coordination core and simulator metrics.
- **`src/`**: Simulator implementations.
- **`tests/`**: Unit tests.
  - `test_runner.cpp`: C++ test cases.
- **`CMakeLists.txt`**: Project CMake build script.
- **`main.cpp`**: Main driver showing oneAPI queue dispatch and execution.
- **`DESIGN.md`**: In-depth hardware architectural specifications.

## Prerequisites

- C++17 compatible compiler (GCC, Clang, or MSVC cl).
- CMake (version 3.15 or newer).

## Building the Project

Run standard CMake build commands:

```bash
# Configure the project files
cmake -B build

# Build the simulator and test binaries
cmake --build build
```

## Running the Simulator

Run the compiled executable to simulate running a oneAPI workload on Panther Lake:

```bash
# Windows
.\build\panther_lake_sim.exe

# Linux/macOS
./build/panther_lake_sim
```

## Running the Unit Tests

Verify the performance models compile and compute metrics properly:

```bash
# Windows
.\build\test_runner.exe

# Linux/macOS
./build/test_runner
```
