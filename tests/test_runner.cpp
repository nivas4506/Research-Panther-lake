#include <iostream>
#include <cassert>
#include "panther_lake/sycl_mock.hpp"
#include "panther_lake/cores.hpp"
#include "panther_lake/gpu.hpp"
#include "panther_lake/npu.hpp"

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

int main() {
    test_sycl_mock();
    test_cpu_cores();
    test_gpu();
    test_npu();
    return 0;
}
