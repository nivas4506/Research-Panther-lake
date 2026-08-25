#include <iostream>
#include <cassert>
#include "panther_lake/sycl_mock.hpp"

void test_sycl_mock() {
    sycl::device cpu_dev(sycl::device_type::cpu);
    sycl::device gpu_dev(sycl::device_type::gpu);
    sycl::queue q(gpu_dev);
    
    assert(q.get_device().get_type() == sycl::device_type::gpu);
    std::cout << "test_sycl_mock PASS" << std::endl;
}

int main() {
    test_sycl_mock();
    return 0;
}
