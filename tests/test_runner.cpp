#include <iostream>
#include <cassert>
#include "panther_lake/sycl_mock.hpp"
#include "panther_lake/cores.hpp"
#include "panther_lake/gpu.hpp"
#include "panther_lake/npu.hpp"
#include "panther_lake/memory.hpp"
#include "panther_lake/fabric.hpp"
#include "panther_lake/platform.hpp"
#include "panther_lake/simulator.hpp"

void test_sycl_mock() {
    sycl::device cpu_dev(sycl::device_type::cpu);
    sycl::device gpu_dev(sycl::device_type::gpu);
    sycl::queue q(gpu_dev);
    
    assert(q.get_device().get_type() == sycl::device_type::gpu);
    std::cout << "test_sycl_mock PASS" << std::endl;
}

void test_cpu_cores() {
    panther_lake::CougarCoveCore p_core(0);
    panther_lake::DarkmontCore e_core(1, false);
    
    unsigned long long p_cycles = p_core.execute(10000, 2000);
    unsigned long long e_cycles = e_core.execute(10000, 2000);
    
    assert(p_cycles > 0);
    assert(e_cycles > 0);
    assert(p_core.get_power() > 0.0);
    assert(e_core.get_power() > 0.0);
    std::cout << "test_cpu_cores PASS" << std::endl;
}

void test_xe3_gpu_core() {
    panther_lake::Xe3GpuCore gpu_core(0, 1.6);
    unsigned long long cycles = gpu_core.execute(100000, 20000);
    assert(cycles > 0);
    assert(gpu_core.get_power() > 0.0);
    std::cout << "test_xe3_gpu_core PASS" << std::endl;
}

void test_gpu() {
    panther_lake::Xe3GPU gpu(12, 2.5);
    unsigned long long cycles = gpu.execute(1000000, 50000);
    assert(cycles > 0);
    assert(gpu.get_power() > 0.0);
    assert(gpu.get_gflops() > 0.0);
    std::cout << "test_gpu PASS" << std::endl;
}

void test_npu() {
    panther_lake::NPU5 npu(1.8);
    unsigned long long cycles = npu.execute_matmul(128, 128, 128);
    assert(cycles > 0);
    assert(npu.get_power() > 0.0);
    assert(npu.get_max_tops() >= 40.0);
    std::cout << "test_npu PASS" << std::endl;
}

void test_memory() {
    panther_lake::LPDDR5XController memory(9600.0, 2);
    assert(memory.get_peak_bandwidth_gbs() > 150.0);
    
    panther_lake::SystemLevelCache slc(24);
    auto [hit, cycles] = slc.access(0x1000);
    assert(cycles > 0);
    std::cout << "test_memory PASS" << std::endl;
}

void test_fabric() {
    panther_lake::FoverosInterconnect fabric;
    int lat = fabric.route("ComputeTile", "GraphicsTile", 64);
    assert(lat > 0);
    assert(fabric.get_total_bytes_routed() == 64);
    std::cout << "test_fabric PASS" << std::endl;
}

void test_platform() {
    panther_lake::PlatformControllerTile pct;
    int cycles = pct.simulate_io(1000000, 500000);
    assert(cycles > 0);
    assert(pct.get_power() > 0.0);
    assert(pct.calculate_dvfs_voltage(5.0) > 0.7);
    std::cout << "test_platform PASS" << std::endl;
}

void test_simulator() {
    panther_lake::PantherLakeProcessor processor(true);
    panther_lake::Workload workload;
    workload.cpu_instructions = 500000;
    workload.cpu_mem_ops = 100000;
    workload.gpu_ops = 100000;
    workload.gpu_matrix_ops = 20000;
    workload.npu_m = 128; workload.npu_n = 128; workload.npu_k = 128;
    workload.pcie_bytes = 50000;
    workload.tb_bytes = 10000;
    
    auto res = processor.run_workload(workload);
    assert(res.total_cycles > 0);
    assert(res.avg_power_watts > 0.0);
    assert(res.total_energy_joules > 0.0);
    std::cout << "test_simulator PASS" << std::endl;
}

void test_execution_core_polymorphism() {
    std::vector<std::unique_ptr<panther_lake::ExecutionCore>> cores;
    cores.push_back(std::make_unique<panther_lake::CougarCoveCore>(0));
    cores.push_back(std::make_unique<panther_lake::DarkmontCore>(1, false));
    
    assert(cores[0]->get_frequency_ghz() == 5.0);
    assert(cores[1]->get_frequency_ghz() == 3.5);
    std::cout << "test_execution_core_polymorphism PASS" << std::endl;
}

void test_dynamic_bandwidth_and_power_limits() {
    panther_lake::PantherLakeProcessor processor(true); // High power config
    panther_lake::Workload workload;
    workload.cpu_instructions = 10000000ULL;
    workload.cpu_mem_ops = 2000000ULL;
    workload.gpu_ops = 800000000ULL;
    workload.gpu_matrix_ops = 200000000ULL;
    workload.npu_m = 2048;
    workload.npu_n = 2048;
    workload.npu_k = 2048;
    
    auto res = processor.run_workload(workload);
    
    // Assert SoC power is capped at 35W
    assert(res.avg_power_watts <= 35.5);
    std::cout << "test_dynamic_bandwidth_and_power_limits PASS" << std::endl;
}

int main() {
    test_sycl_mock();
    test_cpu_cores();
    test_xe3_gpu_core();
    test_execution_core_polymorphism();
    test_gpu();
    test_npu();
    test_memory();
    test_fabric();
    test_platform();
    test_simulator();
    test_dynamic_bandwidth_and_power_limits();
    return 0;
}
