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
    bool battery_saver = false;
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
