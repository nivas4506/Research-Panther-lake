#include "panther_lake/simulator.hpp"
#include <algorithm>
#include <cstdlib>
#include <cmath>

namespace panther_lake {

PantherLakeProcessor::PantherLakeProcessor(bool high_power_config)
    : high_power_config_(high_power_config),
      gpu_(high_power_config ? 12 : 4, high_power_config ? 1.6 : 1.2),
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
    // 1. Dynamic Bandwidth Partitioning Setup
    double gpu_bw_fraction = 0.15;
    double npu_bw_fraction = 0.15;
    
    bool npu_active = (workload.npu_m > 0 && workload.npu_n > 0 && workload.npu_k > 0);
    bool gpu_active = (workload.gpu_ops > 0 || workload.gpu_matrix_ops > 0);
    
    if (npu_active && !gpu_active) {
        npu_bw_fraction = 0.85;
        gpu_bw_fraction = 0.05;
    } else if (gpu_active && !npu_active) {
        gpu_bw_fraction = 0.85;
        npu_bw_fraction = 0.05;
    } else if (gpu_active && npu_active) {
        // Contention / overlap case
        gpu_bw_fraction = 0.60;
        npu_bw_fraction = 0.30;
    }
    
    double peak_bw = memory_ctrl_.get_peak_bandwidth_gbs();
    double gpu_allocated_bw = peak_bw * gpu_bw_fraction;
    double npu_allocated_bw = peak_bw * npu_bw_fraction;

    // Estimate memory transfer times based on workload size (FP16 is 2.0 bytes; INT4 is 0.5 bytes in battery saver)
    double precision_multiplier = workload.battery_saver ? 0.5 : 2.0;
    double gpu_data_bytes = (workload.gpu_ops * precision_multiplier);
    double npu_data_bytes = (static_cast<double>(workload.npu_m) * workload.npu_n * workload.npu_k * precision_multiplier);
    
    // Convert transfer seconds to Foveros base cycles (2.0 GHz clock)
    unsigned long long gpu_mem_transfer_cycles = static_cast<unsigned long long>(
        (gpu_data_bytes / (gpu_allocated_bw * 1000000000.0)) * 2000000000.0
    );
    unsigned long long npu_mem_transfer_cycles = static_cast<unsigned long long>(
        (npu_data_bytes / (npu_allocated_bw * 1000000000.0)) * 2000000000.0
    );

    // 2. Capture initial core frequencies for DVFS scaling
    double initial_p_freq = p_cores_[0]->get_frequency_ghz();
    double initial_e_freq = e_cores_.empty() ? 0.0 : e_cores_[0]->get_frequency_ghz();
    double initial_gpu_freq = gpu_.get_gpu_cores()[0]->get_frequency_ghz();
    
    double p_freq = initial_p_freq;
    double e_freq = initial_e_freq;
    double gpu_freq = initial_gpu_freq;
    
    double target_tdp = workload.battery_saver ? 15.0 : 35.0; // Dynamic 15W/35W TDP Wall
    
    if (workload.battery_saver) {
        for (auto& core : p_cores_) {
            core->set_gated(true);
        }
    }
    
    double total_power = 0.0;
    
    // Dynamic Voltage & Frequency Scaling Throttling Loop
    while (true) {
        double cpu_power = 0.0;
        for (auto& core : p_cores_) cpu_power += core->get_power();
        for (auto& core : e_cores_) cpu_power += core->get_power();
        for (auto& core : lp_cores_) cpu_power += core->get_power();
        
        double gpu_power = gpu_.get_power();
        double npu_power = npu_.get_power();
        double pct_power = pct_.get_power();
        
        total_power = cpu_power + gpu_power + npu_power + pct_power;
        
        // Stop scaling if power is within TDP envelope or we hit minimum clocks
        if (total_power <= target_tdp || (p_freq <= 1.5 && gpu_freq <= 0.8)) {
            break;
        }
        
        // Step down frequencies dynamically
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

    // 3. Execution Simulation
    unsigned long long cpu_cycles = 0;
    
    if (workload.battery_saver) {
        // P-cores are gated. Route CPU workload to E-cores and LP-cores
        if (high_power_config_ && !e_cores_.empty()) {
            unsigned long long e_work = static_cast<unsigned long long>(workload.cpu_instructions * 0.70 / e_cores_.size());
            unsigned long long e_mem = static_cast<unsigned long long>(workload.cpu_mem_ops * 0.70 / e_cores_.size());
            for (auto& core : e_cores_) {
                cpu_cycles = std::max(cpu_cycles, core->execute(e_work, e_mem));
            }
            unsigned long long lp_work = static_cast<unsigned long long>(workload.cpu_instructions * 0.30 / lp_cores_.size());
            unsigned long long lp_mem = static_cast<unsigned long long>(workload.cpu_mem_ops * 0.30 / lp_cores_.size());
            for (auto& core : lp_cores_) {
                cpu_cycles = std::max(cpu_cycles, core->execute(lp_work, lp_mem));
            }
        } else {
            unsigned long long lp_work = static_cast<unsigned long long>(workload.cpu_instructions / lp_cores_.size());
            unsigned long long lp_mem = static_cast<unsigned long long>(workload.cpu_mem_ops / lp_cores_.size());
            for (auto& core : lp_cores_) {
                cpu_cycles = std::max(cpu_cycles, core->execute(lp_work, lp_mem));
            }
        }
    } else {
        // CPU work sharing - Normal Mode
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
    }
    
    unsigned long long gpu_cycles = gpu_.execute(workload.gpu_ops, workload.gpu_matrix_ops) + gpu_mem_transfer_cycles;
    unsigned long long npu_cycles = npu_.execute_matmul(workload.npu_m, workload.npu_n, workload.npu_k) + npu_mem_transfer_cycles;
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
    
    unsigned long long fabric_cycles = fabric_.route("ComputeTile", "GraphicsTile", workload.gpu_ops * 4) +
                        fabric_.route("ComputeTile", "PlatformControllerTile", workload.pcie_bytes + workload.tb_bytes);
                        
    unsigned long long total_cycles = std::max({cpu_cycles, gpu_cycles, npu_cycles, pct_cycles}) + fabric_cycles + slc_access_cycles;
    
    double simulation_time_sec = total_cycles / (2.0 * 1000000000.0);
    double total_energy_joules = total_power * simulation_time_sec;
    
    double cpu_ips = workload.cpu_instructions / simulation_time_sec;
    double gpu_gflops = (workload.gpu_ops + workload.gpu_matrix_ops * 2) / (simulation_time_sec * 1000000000.0);
    double npu_tops = (static_cast<double>(workload.npu_m) * workload.npu_n * workload.npu_k * 2) / (simulation_time_sec * 1000000000000.0);
    
    double peak_temp_c = 35.0 + 1.2 * total_power;
    
    // 4. Restore base frequencies and gating status for subsequent workload dispatches
    for (auto& core : p_cores_) {
        core->set_frequency_ghz(initial_p_freq);
        core->set_gated(false);
    }
    for (auto& core : e_cores_) core->set_frequency_ghz(initial_e_freq);
    for (auto& core : gpu_.get_gpu_cores()) core->set_frequency_ghz(initial_gpu_freq);
    
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
