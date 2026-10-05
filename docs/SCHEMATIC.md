# Intel Panther Lake Modified Chip Architecture & Schematic Diagram

This document provides a detailed block-level schematic and control-flow diagram for the modified Intel Panther Lake processor (Core Ultra Series 3) simulated in the C++ oneAPI simulation framework.

---

## 1. Physical Chiplet Layout & Schematic Design

Below is the logical block-level blueprint schematic for the modified disaggregated chip packaging:

![Intel Panther Lake SoC Architecture Modified Schematic Design](../modified_chip_schematic.jpg)

---

## 2. Chiplet Logical Block Schematic

The diagram below maps out the tiles, internal units, local buses, and interconnect links (Foveros 3D packaging and memory channels):

```mermaid
graph TB
    %% Styling definitions
    classDef compute fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef graphics fill:#efebe9,stroke:#4e342e,stroke-width:2px;
    classDef platform fill:#efe8ff,stroke:#4a148c,stroke-width:2px;
    classDef memory fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef bus fill:#fffde7,stroke:#f57f17,stroke-width:2px,stroke-dasharray: 5 5;

    %% Compute Tile
    subgraph ComputeTile ["Compute Tile (Intel 18A)"]
        direction TB
        PCores["4x Cougar Cove P-Cores<br/>(5.0 GHz nominal)<br/>[Power Gated in Battery-Saver Mode]"]
        ECores["8x Darkmont E-Cores<br/>(3.5 GHz nominal)"]
        LPECores["4x Darkmont LP E-Cores<br/>(2.5 GHz nominal)"]
        NPU["NPU 5.0 Accelerator<br/>(4x Neural Compute Engines @ 1.8 GHz)"]
        SLC["24MB System Level Cache (SLC)"]
        CompRing["Compute Ring Bus / Cache Coherency Fabric"]
        
        PCores --- CompRing
        ECores --- CompRing
        LPECores --- CompRing
        NPU --- CompRing
        SLC --- CompRing
    end
    class ComputeTile compute;

    %% Graphics Tile
    subgraph GraphicsTile ["Graphics Tile (TSMC N3E / Intel 3)"]
        direction TB
        GfxCores["12x Xe3 GPU Cores<br/>(Celestial Architecture @ 1.6 GHz nominal)<br/>(8x Vector Engines & 8x XMX Matrix Engines per Core)"]
        GfxBus["Graphics Tile Local Interconnect"]
        
        GfxCores --- GfxBus
    end
    class GraphicsTile graphics;

    %% Platform Controller Tile
    subgraph PlatformTile ["Platform Controller Tile (TSMC N6)"]
        direction TB
        PCIe["PCIe Gen 5 Controller<br/>(Up to 63 GB/s)"]
        TB5["Thunderbolt 5 Controller<br/>(Up to 10 GB/s)"]
        DVFS["DVFS Regulator<br/>(Throttles Core Frequencies)"]
        PlatBus["PCT System Interconnect"]
        
        PCIe --- PlatBus
        TB5 --- PlatBus
        DVFS --- PlatBus
    end
    class PlatformTile platform;

    %% Memory Subsystem
    subgraph MemorySubsystem ["Memory Subsystem"]
        direction LR
        MC["LPDDR5X Memory Controller<br/>(Dynamic Bandwidth Partitioner)"]
        DRAM["128GB LPDDR5X-9600 RAM<br/>(Dual Channel, 153.6 GB/s Peak)"]
        
        MC === DRAM
    end
    class MemorySubsystem memory;
    class MC,DRAM memory;

    %% Foveros 3D Packaging Interconnects
    CompRing <== "Foveros 3D Link<br/>(256.0 GB/s)" ==> GfxBus
    CompRing <== "Foveros 3D Link<br/>(128.0 GB/s)" ==> PlatBus
    GfxBus <== "Foveros 3D Link<br/>(64.0 GB/s)" ==> PlatBus

    %% Memory bus connections
    CompRing -. "Shared Memory Bus" .-> MC
    GfxBus -. "Shared Memory Bus" .-> MC
    class CompRing,GfxBus,PlatBus bus;
```

---

## 3. Dynamic Control Logic & Simulation Flow

The flowchart below traces the simulator's control sequences during workload dispatch, illustrating how DVFS, Memory Partitioning, and Battery-Saver logic interact.

```mermaid
flowchart TD
    %% Styling
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef process fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px;
    classDef startend fill:#eceff1,stroke:#37474f,stroke-width:2px;
    classDef loop fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    
    Start([Start Workload Simulation]) --> CheckSaver{Battery-Saver Active?}
    class CheckSaver decision;
    
    %% Battery-Saver Path
    CheckSaver -- Yes --> GatePCores["Power-Gate P-Cores<br/>(Clamp dynamic/leakage power to 0W)"]
    GatePCores --> RouteULP["Route CPU Workload:<br/>70% to E-Cores, 30% to LP E-Cores"]
    RouteULP --> QuantizeINT4["Scale Model Precision to INT4<br/>(Quantize weights, reduce footprint by 4x)"]
    QuantizeINT4 --> Set15WTDP["Set Target Package TDP = 15.0W"]
    class GatePCores,RouteULP,QuantizeINT4,Set15WTDP process;
    
    %% Normal Path
    CheckSaver -- No --> RouteNormal["Route CPU Workload:<br/>70% to P-Cores, 30% to E-Cores (High Power)"]
    RouteNormal --> RunFP16["Run Native FP16 Model Precision<br/>(Full memory footprint)"]
    RunFP16 --> Set35WTDP["Set Target Package TDP = 35.0W"]
    class RouteNormal,RunFP16,Set35WTDP process;
    
    %% Dynamic Bandwidth Partitioning
    Set15WTDP & Set35WTDP --> DetectPhase{Detect Workload Phase}
    class DetectPhase decision;
    
    DetectPhase -- "NPU Active Only" --> AllocNPU["Partition Bandwidth:<br/>85% NPU, 15% CPU/GPU"]
    DetectPhase -- "GPU Active Only" --> AllocGPU["Partition Bandwidth:<br/>85% GPU, 15% CPU/NPU"]
    DetectPhase -- "GPU & NPU Both Active" --> ContentionSplit["Partition Bandwidth:<br/>60% GPU, 30% NPU, 10% CPU/others"]
    class AllocNPU,AllocGPU,ContentionSplit process;
    
    %% DVFS Power Throttling Loop
    AllocNPU & AllocGPU & ContentionSplit --> EstPower["Calculate Total Package Power Draw<br/>(Sum CPU, GPU, NPU, Platform, Cache)"]
    class EstPower loop;
    
    EstPower --> CheckTDP{Total Power > TDP Limit?}
    class CheckTDP decision;
    
    CheckTDP -- Yes --> CheckFloor{Frequencies > Minimum Floor?<br/>(P-core > 1.5G, E-core > 1.0G, GPU > 0.8G)}
    class CheckFloor decision;
    
    CheckFloor -- Yes --> ScaleFreq["DVFS Power Throttling:<br/>Step down CPU & GPU frequencies by 0.1 GHz"]
    ScaleFreq --> EstPower
    class ScaleFreq process;
    
    CheckFloor -- No --> RunSim["Execute Cycle-Approximate Simulation Step"]
    CheckTDP -- No --> RunSim
    class RunSim process;
    
    RunSim --> ResetStates["Reset active power metrics to nominal<br/>via reset_active_power()"]
    ResetStates --> End([End Workload Simulation])
    class ResetStates process;
    class Start,End startend;
```
