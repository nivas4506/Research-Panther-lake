#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <algorithm>
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

void print_dashboard(int step, const std::string& phase_name, double sim_time_ms, double power_w, double temp_c,
                     double cpu_util, double gpu_util, double npu_util, unsigned long long bytes_routed) {
    // ANSI Escape Code: Clear screen and home cursor
    std::cout << "\033[H\033[J";
    std::cout << "=====================================================================" << std::endl;
    std::cout << "   Intel Panther Lake - oneAPI C++ Real-Time Telemetry Dashboard     " << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << "  [Current Phase]   " << phase_name << " (" << step << "/10)" << std::endl;
    std::cout << "  [Simulated Time]  " << std::fixed << std::setprecision(2) << sim_time_ms << " ms / 522.40 ms" << std::endl;
    std::cout << "  [Progress Bar]    [" << get_utilization_bar(step / 10.0, 30) << "] " << step * 10 << "%" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "  TILE HEALTH & PHYSICAL METRICS:" << std::endl;
    std::cout << "    SoC Active Power Draw:  " << std::setprecision(2) << power_w << " Watts  [CLAMPED AT 35W TDP]" << std::endl;
    std::cout << "    Compute Tile Temp:      " << std::setprecision(1) << temp_c << " C  ";
    if (temp_c > 90.0) {
        std::cout << "\033[1;31m[HOT - THERMAL THROTTLING ACTIVATED]\033[0m";
    } else if (temp_c > 75.0) {
        std::cout << "\033[1;33m[WARM]\033[0m";
    } else {
        std::cout << "\033[1;32m[OPTIMAL]\033[0m";
    }
    std::cout << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "  COMPUTATIONAL SUB-UNITS UTILIZATION (FIRST-CLASS CORES):" << std::endl;
    std::cout << "    Cougar Cove (P-Cores):  [" << get_utilization_bar(cpu_util, 20) << "] " << std::setprecision(1) << cpu_util * 100.0 << "%" << std::endl;
    std::cout << "    Blackwell GPU Cores:    [" << get_utilization_bar(gpu_util, 20) << "] " << std::setprecision(1) << gpu_util * 100.0 << "%" << std::endl;
    std::cout << "    NPU 5 Tensor Engines:   [" << get_utilization_bar(npu_util, 20) << "] " << std::setprecision(1) << npu_util * 100.0 << "%" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;
    std::cout << "  UNIFIED MEMORY (LPDDR5X-9600) & FOVEROS FABRIC CONTROLS:" << std::endl;
    std::cout << "    Volume Routed:          " << std::setprecision(3) << bytes_routed / 1000000000.0 << " GB / 81.0 GB (40B Model FP16)" << std::endl;
    std::cout << "    Routing Bus Bandwidth:  256.0 GB/s link (Compute <-> Graphics)" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout.flush();
}

int main() {
    // Initial clear to prepare screen
    std::cout << "\033[H\033[J";
    
    // Instantiate processor in High Power config
    panther_lake::PantherLakeProcessor processor(true);
    
    // Simulate oneAPI program submissions targeting the integrated NVIDIA GPU
    sycl::device dev(sycl::device_type::gpu);
    sycl::queue q(dev);
    
    struct SimulationPhase {
        std::string name;
        double cpu_util;
        double gpu_util;
        double npu_util;
        unsigned long long bytes_transferred;
    };
    
    std::vector<SimulationPhase> phases = {
        {"oneAPI Context Setup & Device Query", 0.05, 0.00, 0.00, 1024},
        {"128GB Unified Memory Initialization", 0.15, 0.00, 0.00, 4096},
        {"Loading 40B FP16 Model Weights (80GB)", 0.20, 0.10, 0.00, 80000000000ULL},
        {"Prompt Embedding Dispatch & Cache Map", 0.35, 0.05, 0.10, 80100000000ULL},
        {"Prompt Processing Layers (NPU 5.0 Active)", 0.10, 0.05, 0.95, 80500000000ULL},
        {"KV Cache Allocation & Memory Tagging", 0.25, 0.10, 0.20, 80600000000ULL},
        {"Token Decode Wavefront (Blackwell GPU)", 0.15, 0.85, 0.05, 80800000000ULL},
        {"Autoregressive Token Gen Loop (GPU Core)", 0.10, 0.98, 0.00, 81000000000ULL},
        {"Device-to-Host Output Stream Sync", 0.20, 0.05, 0.00, 81000005000ULL},
        {"oneAPI Context Clean & Synchronize", 0.05, 0.00, 0.00, 81000005000ULL}
    };
    
    // Workload dimensions modeling 40B FP16 generation steps
    panther_lake::Workload workload;
    workload.cpu_instructions = 500000000ULL; // 500M instructions
    workload.cpu_mem_ops = 100000000ULL;
    workload.gpu_ops = 80000000000ULL;        // 80G vector floating-point operations
    workload.gpu_matrix_ops = 40000000000ULL; // 40G matrix ops
    workload.npu_m = 2048;
    workload.npu_n = 2048;
    workload.npu_k = 2048;
    workload.pcie_bytes = 1000000ULL;         // Low PCIe copies (on-package integrated)
    workload.tb_bytes = 0ULL;
    
    double sim_time_accumulated_ms = 0.0;
    double current_temp = 35.0;
    
    // Execute live dashboard step-by-step (updating metrics to reflect 35W TDP cap and DVFS)
    for (size_t i = 0; i < phases.size(); ++i) {
        int step = i + 1;
        const auto& phase = phases[i];
        
        // Calculate estimated dynamic power under 35W clamp
        double raw_power = (phase.cpu_util * 15.0) + (phase.gpu_util * 45.0) + (phase.npu_util * 6.0) + 1.5;
        double clamped_power = std::min(raw_power, 35.0); // Enforce 35W limit
        
        current_temp = 35.0 + 1.2 * clamped_power;
        sim_time_accumulated_ms = (step / 10.0) * 522.40;
        
        print_dashboard(step, phase.name, sim_time_accumulated_ms, clamped_power, current_temp,
                        phase.cpu_util, phase.gpu_util, phase.npu_util, phase.bytes_transferred);
        
        // Dynamic wait simulating real-world refresh rates
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    // Perform final high-precision simulator run
    auto final_result = processor.run_workload(workload);
    
    std::cout << "\n================= FINAL SIMULATOR LOGS =================" << std::endl;
    std::cout << "  Workload Scenario:        40 Billion Parameter FP16 Model" << std::endl;
    std::cout << "  Execution completed in:   " << final_result.simulation_time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "  Total cycles elapsed:     " << final_result.total_cycles << " cycles" << std::endl;
    std::cout << "  Average SoC power:        " << final_result.avg_power_watts << " W  (Power Clamp: 35W)" << std::endl;
    std::cout << "  Total dynamic energy:     " << final_result.total_energy_joules * 1000.0 << " mJ" << std::endl;
    std::cout << "  Estimated peak core temp: " << final_result.peak_temp_c << " C" << std::endl;
    std::cout << "  oneAPI CPU Throughput:    " << final_result.cpu_ips / 1000000.0 << " MIPS" << std::endl;
    std::cout << "  oneAPI GPU Performance:   " << final_result.gpu_gflops << " GFLOPS (Integrated Blackwell)" << std::endl;
    std::cout << "  oneAPI NPU Performance:   " << final_result.npu_tops << " TOPS" << std::endl;
    std::cout << "========================================================\n" << std::endl;
    
    return 0;
}
