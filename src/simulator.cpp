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
