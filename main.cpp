#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
#include <string>
#include <vector>
#include "panther_lake/sycl_mock.hpp"
#include "panther_lake/simulator.hpp"

std::string get_utilization_bar(double util, int width = 20) {
    std::string bar = "";
    int pos = static_cast<int>(util * width);
    for (int i = 0; i < width; ++i) {
        if (i < pos) bar += "=";
        else if (i == pos) bar += ">";
        else bar += " ";
    }
    return bar;
}

void print_dashboard(const std::string& arch_title, int step, const std::string& phase_name, 
                     double sim_time_ms, double total_sim_ms, double power_w, double temp_c,
                     double cpu_util, double gpu_util, double npu_util, unsigned long long bytes_routed, 
                     double tdp_cap = 35.0, bool is_unthrottled = false) {
    // ANSI Escape Code: Clear screen and home cursor
    std::cout << "\033[H\033[J";
    std::cout << "================================================================================" << std::endl;
    std::cout << "   " << arch_title << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "  [Current Phase]   " << phase_name << " (" << step << "/10)" << std::endl;
    std::cout << "  [Simulated Time]  " << std::fixed << std::setprecision(2) << sim_time_ms << " ms / " << total_sim_ms << " ms" << std::endl;
    std::cout << "  [Progress Bar]    [" << get_utilization_bar(step / 10.0, 30) << "] " << step * 10 << "%" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "  TILE HEALTH & PHYSICAL METRICS:" << std::endl;
    std::cout << "    SoC Active Power Draw:  " << std::setprecision(2) << power_w << " Watts  ";
    if (is_unthrottled) {
        std::cout << "[RAW UNTHROTTLED / NO DVFS]" << std::endl;
    } else {
        std::cout << "[CLAMPED AT " << tdp_cap << "W TDP]" << std::endl;
    }
    std::cout << "    Compute Tile Temp:      " << std::setprecision(1) << temp_c << " C  ";
    if (temp_c > 90.0) {
        std::cout << "\033[1;31m[HOT - THERMAL THROTTLING ACTIVATED]\033[0m";
    } else if (temp_c > 75.0) {
        std::cout << "\033[1;33m[WARM]\033[0m";
    } else {
        std::cout << "\033[1;32m[OPTIMAL]\033[0m";
    }
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "  COMPUTATIONAL SUB-UNITS UTILIZATION (FIRST-CLASS CORES):" << std::endl;
    std::cout << "    Cougar Cove (P-Cores):  [" << get_utilization_bar(cpu_util, 20) << "] " << std::setprecision(1) << cpu_util * 100.0 << "%" << std::endl;
    std::cout << "    Intel Xe3 GPU Cores:    [" << get_utilization_bar(gpu_util, 20) << "] " << std::setprecision(1) << gpu_util * 100.0 << "%" << std::endl;
    std::cout << "    NPU 5 Tensor Engines:   [" << get_utilization_bar(npu_util, 20) << "] " << std::setprecision(1) << npu_util * 100.0 << "%" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "  UNIFIED MEMORY (LPDDR5X-9600) & FOVEROS FABRIC CONTROLS:" << std::endl;
    std::cout << "    Volume Routed:          " << std::setprecision(3) << bytes_routed / 1000000000.0 << " GB / 81.0 GB" << std::endl;
    std::cout << "    Routing Bus Bandwidth:  256.0 GB/s link (Compute <-> Graphics)" << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout.flush();
}

struct SimulationPhase {
    std::string name;
    double cpu_util;
    double gpu_util;
    double npu_util;
    unsigned long long bytes_transferred;
};

const std::vector<SimulationPhase> g_phases = {
    {"oneAPI Context Setup & Device Query", 0.05, 0.00, 0.00, 1024},
    {"128GB Unified Memory Initialization", 0.15, 0.00, 0.00, 4096},
    {"Loading 40B FP16 Model Weights (80GB)", 0.20, 0.10, 0.00, 80000000000ULL},
    {"Prompt Embedding Dispatch & Cache Map", 0.35, 0.05, 0.10, 80100000000ULL},
    {"Prompt Processing Layers (NPU 5.0 Active)", 0.10, 0.05, 0.95, 80500000000ULL},
    {"KV Cache Allocation & Memory Tagging", 0.25, 0.10, 0.20, 80600000000ULL},
    {"Token Decode Wavefront (Intel Xe3 GPU)", 0.15, 0.85, 0.05, 80800000000ULL},
    {"Autoregressive Token Gen Loop (GPU Core)", 0.10, 0.98, 0.00, 81000000000ULL},
    {"Device-to-Host Output Stream Sync", 0.20, 0.05, 0.00, 81000005000ULL},
    {"oneAPI Context Clean & Synchronize", 0.05, 0.00, 0.00, 81000005000ULL}
};

void run_sim_visual(const std::string& title, double tdp_cap, bool is_unthrottled, 
                    bool is_saver, int delay_ms, double total_sim_ms) {
    if (delay_ms <= 0) return;
    for (size_t i = 0; i < g_phases.size(); ++i) {
        int step = i + 1;
        const auto& phase = g_phases[i];
        
        double cpu_u = phase.cpu_util;
        if (is_saver) cpu_u *= 0.4;
        
        double raw_power = 0.0;
        if (is_saver) {
            raw_power = (cpu_u * 5.0) + (phase.gpu_util * 12.0) + (phase.npu_util * 2.0) + 0.8;
        } else {
            raw_power = (cpu_u * 15.0) + (phase.gpu_util * 45.0) + (phase.npu_util * 6.0) + 1.5;
        }
        
        double display_power = is_unthrottled ? raw_power : std::min(raw_power, tdp_cap);
        double current_temp = 35.0 + 1.2 * display_power;
        double sim_time_accumulated = (step / 10.0) * total_sim_ms;
        
        unsigned long long bytes = phase.bytes_transferred;
        if (is_saver && step > 2) bytes /= 4;
        
        print_dashboard(title, step, phase.name, sim_time_accumulated, total_sim_ms,
                        display_power, current_temp, cpu_u, phase.gpu_util, phase.npu_util,
                        bytes, tdp_cap, is_unthrottled);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
}

int main(int argc, char* argv[]) {
    bool run_original = true;
    bool run_modified = true;
    int delay_ms = 150;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--original" || arg == "-o") {
            run_original = true;
            run_modified = false;
        } else if (arg == "--modified" || arg == "-m") {
            run_original = false;
            run_modified = true;
        } else if (arg == "--all" || arg == "-a") {
            run_original = true;
            run_modified = true;
        } else if (arg == "--quick" || arg == "--no-delay" || arg == "-q") {
            delay_ms = 0;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Intel Panther Lake & Modified Chip oneAPI C++ Simulator\n"
                      << "Usage: panther_lake_sim.exe [options]\n\n"
                      << "Options:\n"
                      << "  --all, -a         Run both Original Baseline and Modified Chip simulations (default)\n"
                      << "  --original, -o    Run Original Panther Lake Baseline simulation only\n"
                      << "  --modified, -m    Run Modified Chip simulations only (Normal 35W & Battery-Saver 15W)\n"
                      << "  --quick, -q       Execute simulations instantly without animation delay\n"
                      << "  --help, -h        Display this help message\n";
            return 0;
        }
    }
    
    // Initial clear
    std::cout << "\033[H\033[J";
    
    panther_lake::PantherLakeProcessor processor(true);
    
    // Setup standard 40B LLM Workload
    panther_lake::Workload base_workload;
    base_workload.cpu_instructions = 500000000ULL; // 500M instructions
    base_workload.cpu_mem_ops = 100000000ULL;
    base_workload.gpu_ops = 80000000000ULL;        // 80G vector FLOPs
    base_workload.gpu_matrix_ops = 40000000000ULL; // 40G matrix ops
    base_workload.npu_m = 2048;
    base_workload.npu_n = 2048;
    base_workload.npu_k = 2048;
    base_workload.pcie_bytes = 1000000ULL;
    base_workload.tb_bytes = 0ULL;
    
    panther_lake::SimulationResult orig_result{};
    panther_lake::SimulationResult mod_normal_result{};
    panther_lake::SimulationResult mod_saver_result{};
    
    // ---------------------------------------------------------
    // 1. ORIGINAL PANTHER LAKE BASELINE SIMULATION
    // ---------------------------------------------------------
    if (run_original) {
        std::cout << ">>> Launching Original Panther Lake Baseline Simulation... <<<\n" << std::endl;
        run_sim_visual("Original Panther Lake Baseline (Fixed Clocks, Static 50/50 BW, FP16)", 
                       41.2, true, false, delay_ms, 3750.0);
        
        auto orig_wl = base_workload;
        orig_wl.is_original_baseline = true;
        orig_result = processor.run_workload(orig_wl);
        
        if (run_modified && delay_ms > 0) {
            std::cout << "\n>>> Original Simulation Complete. Proceeding to Modified Chip Simulations... <<<" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    // ---------------------------------------------------------
    // 2. MODIFIED CHIP - NORMAL MODE (35W TDP)
    // ---------------------------------------------------------
    if (run_modified) {
        run_sim_visual("Modified Chip (AI Superchip) - Normal Mode (DVFS, Dynamic BW, 35W TDP)", 
                       35.0, false, false, delay_ms, 3170.1);
        
        auto mod_normal_wl = base_workload;
        mod_normal_wl.is_original_baseline = false;
        mod_normal_wl.battery_saver = false;
        mod_normal_result = processor.run_workload(mod_normal_wl);
        
        if (delay_ms > 0) {
            std::cout << "\n>>> Transitioning to Modified Chip: Battery-Saver Mode (15W TDP)... <<<" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        // ---------------------------------------------------------
        // 3. MODIFIED CHIP - BATTERY-SAVER MODE (15W TDP)
        // ---------------------------------------------------------
        run_sim_visual("Modified Chip (AI Superchip) - Battery-Saver Mode (Gated P-Cores, INT4, 15W TDP)", 
                       15.0, false, true, delay_ms, 2093.0);
        
        auto mod_saver_wl = base_workload;
        mod_saver_wl.is_original_baseline = false;
        mod_saver_wl.battery_saver = true;
        mod_saver_result = processor.run_workload(mod_saver_wl);
    }
    
    // ---------------------------------------------------------
    // FINAL COMPREHENSIVE OUTPUT MATRIX
    // ---------------------------------------------------------
    std::cout << "\033[H\033[J";
    std::cout << "=========================================================================================" << std::endl;
    std::cout << "           PANTHER LAKE vs MODIFIED CHIP ARCHITECTURE SIMULATION RESULTS                 " << std::endl;
    std::cout << "=========================================================================================" << std::endl;
    
    if (run_original && run_modified) {
        std::cout << "  METRIC                      ORIGINAL BASELINE     MODIFIED (35W TDP)    MODIFIED (15W TDP)\n";
        std::cout << "  ---------------------------------------------------------------------------------------\n";
        std::cout << "  Architecture / Design:      Panther Lake Gen-1    AI Superchip Normal   AI Superchip ULP\n";
        std::cout << "  Frequency / DVFS Policy:    Fixed (No DVFS)       Adaptive 35W DVFS     Power-Gated 15W DVFS\n";
        std::cout << "  Memory Partitioning:        Static 50/50 Split    Dynamic (85% Shift)   Dynamic (85% Shift)\n";
        std::cout << "  Precision & Footprint:      FP16 (80.0 GB)        FP16 (80.0 GB)        INT4 (20.0 GB)\n";
        std::cout << "  Cougar Cove P-Cores:        Active (5.0 GHz)      Scaled (4.2 GHz)      GATED (0.0W Power)\n";
        std::cout << "  ---------------------------------------------------------------------------------------\n";
        std::cout << "  Execution Time (Latency):   " << std::fixed << std::setprecision(1)
                  << std::setw(8) << orig_result.simulation_time_sec * 1000.0 << " ms        "
                  << std::setw(8) << mod_normal_result.simulation_time_sec * 1000.0 << " ms        "
                  << std::setw(8) << mod_saver_result.simulation_time_sec * 1000.0 << " ms\n";
        std::cout << "  Total Cycles Elapsed:       "
                  << std::setw(11) << orig_result.total_cycles << "           "
                  << std::setw(11) << mod_normal_result.total_cycles << "           "
                  << std::setw(11) << mod_saver_result.total_cycles << " cycles\n";
        std::cout << "  Measured SoC Power Draw:    " << std::setprecision(2)
                  << std::setw(8) << orig_result.avg_power_watts << " W         "
                  << std::setw(8) << mod_normal_result.avg_power_watts << " W         "
                  << std::setw(8) << mod_saver_result.avg_power_watts << " W\n";
        std::cout << "  Total Dynamic Energy:       " << std::setprecision(1)
                  << std::setw(8) << orig_result.total_energy_joules * 1000.0 << " mJ       "
                  << std::setw(8) << mod_normal_result.total_energy_joules * 1000.0 << " mJ       "
                  << std::setw(8) << mod_saver_result.total_energy_joules * 1000.0 << " mJ\n";
        std::cout << "  Peak SoC Operating Temp:    " << std::setprecision(1)
                  << std::setw(8) << orig_result.peak_temp_c << " C         "
                  << std::setw(8) << mod_normal_result.peak_temp_c << " C         "
                  << std::setw(8) << mod_saver_result.peak_temp_c << " C\n";
        std::cout << "  oneAPI CPU Throughput:      " << std::setprecision(1)
                  << std::setw(8) << orig_result.cpu_ips / 1000000.0 << " MIPS       "
                  << std::setw(8) << mod_normal_result.cpu_ips / 1000000.0 << " MIPS       "
                  << std::setw(8) << mod_saver_result.cpu_ips / 1000000.0 << " MIPS\n";
        std::cout << "  oneAPI GPU Performance:     " << std::setprecision(1)
                  << std::setw(8) << orig_result.gpu_gflops << " GFLOPS     "
                  << std::setw(8) << mod_normal_result.gpu_gflops << " GFLOPS     "
                  << std::setw(8) << mod_saver_result.gpu_gflops << " GFLOPS\n";
        std::cout << "  NPU Acceleration Metric:    " << std::setprecision(2)
                  << std::setw(8) << (orig_result.npu_tops * 1000.0) << " GOPS       "
                  << std::setw(8) << (mod_normal_result.npu_tops * 1000.0) << " GOPS       "
                  << std::setw(8) << (mod_saver_result.npu_tops * 1000.0) << " GOPS\n";
    } else if (run_original) {
        std::cout << "  ORIGINAL PANTHER LAKE BASELINE RESULTS (UNTHROTTLED):\n";
        std::cout << "  ---------------------------------------------------------------------------------------\n";
        std::cout << "  Execution Time:             " << std::fixed << std::setprecision(1) << orig_result.simulation_time_sec * 1000.0 << " ms\n";
        std::cout << "  Total Cycles:               " << orig_result.total_cycles << " cycles\n";
        std::cout << "  Average SoC Power:          " << std::setprecision(2) << orig_result.avg_power_watts << " W\n";
        std::cout << "  Total Energy:               " << std::setprecision(1) << orig_result.total_energy_joules * 1000.0 << " mJ\n";
        std::cout << "  Peak Temperature:           " << std::setprecision(1) << orig_result.peak_temp_c << " C\n";
        std::cout << "  CPU Throughput:             " << orig_result.cpu_ips / 1000000.0 << " MIPS\n";
        std::cout << "  GPU Performance:            " << orig_result.gpu_gflops << " GFLOPS\n";
    } else {
        std::cout << "  MODIFIED CHIP (AI SUPERCHIP) RESULTS:\n";
        std::cout << "  ---------------------------------------------------------------------------------------\n";
        std::cout << "  METRIC                      MODIFIED (35W TDP)    MODIFIED (15W TDP)\n";
        std::cout << "  Execution Time:             " << std::fixed << std::setprecision(1)
                  << std::setw(8) << mod_normal_result.simulation_time_sec * 1000.0 << " ms        "
                  << std::setw(8) << mod_saver_result.simulation_time_sec * 1000.0 << " ms\n";
        std::cout << "  Total Cycles:               "
                  << std::setw(11) << mod_normal_result.total_cycles << "           "
                  << std::setw(11) << mod_saver_result.total_cycles << " cycles\n";
        std::cout << "  Average SoC Power:          " << std::setprecision(2)
                  << std::setw(8) << mod_normal_result.avg_power_watts << " W         "
                  << std::setw(8) << mod_saver_result.avg_power_watts << " W\n";
        std::cout << "  Total Energy:               " << std::setprecision(1)
                  << std::setw(8) << mod_normal_result.total_energy_joules * 1000.0 << " mJ       "
                  << std::setw(8) << mod_saver_result.total_energy_joules * 1000.0 << " mJ\n";
        std::cout << "  Peak Temperature:           " << std::setprecision(1)
                  << std::setw(8) << mod_normal_result.peak_temp_c << " C         "
                  << std::setw(8) << mod_saver_result.peak_temp_c << " C\n";
        std::cout << "  CPU Throughput:             " << std::setprecision(1)
                  << std::setw(8) << mod_normal_result.cpu_ips / 1000000.0 << " MIPS       "
                  << std::setw(8) << mod_saver_result.cpu_ips / 1000000.0 << " MIPS\n";
        std::cout << "  GPU Performance:            " << std::setprecision(1)
                  << std::setw(8) << mod_normal_result.gpu_gflops << " GFLOPS     "
                  << std::setw(8) << mod_saver_result.gpu_gflops << " GFLOPS\n";
        std::cout << "  NPU Acceleration:           " << std::setprecision(2)
                  << std::setw(8) << (mod_normal_result.npu_tops * 1000.0) << " GOPS       "
                  << std::setw(8) << (mod_saver_result.npu_tops * 1000.0) << " GOPS\n";
    }
    
    std::cout << "=========================================================================================\n" << std::endl;
    return 0;
}
