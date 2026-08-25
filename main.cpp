#include <iostream>
#include <iomanip>
#include "panther_lake/sycl_mock.hpp"
#include "panther_lake/simulator.hpp"

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "   Intel Panther Lake - oneAPI C++ Architectural Simulator" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    panther_lake::PantherLakeProcessor processor(true);
    
    // Simulate a oneAPI SYCL-like program queue submission
    sycl::device dev(sycl::device_type::gpu);
    sycl::queue q(dev);
    
    std::cout << "Dispatching oneAPI program to: " << q.get_device().get_name() << std::endl;
    
    // Build a mock workload representing the kernel dispatch
    panther_lake::Workload workload;
    workload.cpu_instructions = 100000000ULL;
    workload.cpu_mem_ops = 25000000ULL;
    workload.gpu_ops = 800000000ULL;
    workload.gpu_matrix_ops = 200000000ULL;
    workload.npu_m = 2048;
    workload.npu_n = 2048;
    workload.npu_k = 2048;
    workload.pcie_bytes = 50000000ULL;
    workload.tb_bytes = 10000000ULL;
    
    q.submit([&workload](sycl::handler& h) {
        h.parallel_for<class AIKernel>(workload.gpu_ops, [](size_t idx) {
            // Mock GPU kernel execution body
        });
    });
    
    std::cout << "Simulating hardware execution on Panther Lake processor..." << std::endl;
    auto res = processor.run_workload(workload);
    
    std::cout << "\n================ SIMULATION STATISTICS ================" << std::endl;
    std::cout << "Cycles elapsed:       " << res.total_cycles << " cycles" << std::endl;
    std::cout << "Execution time:       " << res.simulation_time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "Average Power:        " << res.avg_power_watts << " W" << std::endl;
    std::cout << "Total Energy:         " << res.total_energy_joules * 1000.0 << " mJ" << std::endl;
    std::cout << "CPU Throughput:       " << res.cpu_ips / 1000000.0 << " MIPS" << std::endl;
    std::cout << "GPU Performance:      " << res.gpu_gflops << " GFLOPS" << std::endl;
    std::cout << "NPU Performance:      " << res.npu_tops << " TOPS" << std::endl;
    std::cout << "Estimated Temperature: " << res.peak_temp_c << " C" << std::endl;
    std::cout << "=======================================================\n" << std::endl;
    
    return 0;
}
