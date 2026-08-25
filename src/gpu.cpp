#include "panther_lake/gpu.hpp"
#include <algorithm>

namespace panther_lake {

Xe3GPU::Xe3GPU(int xe_cores, double frequency_ghz)
    : xe_cores_(xe_cores), frequency_ghz_(frequency_ghz) {
    for (int i = 0; i < xe_cores_; ++i) {
        gpu_cores_.push_back(std::make_unique<Xe3GpuCore>(i, frequency_ghz_));
    }
}

unsigned long long Xe3GPU::execute(unsigned long long ops, unsigned long long matrix_ops) {
    unsigned long long max_cycles = 0;
    
    // Distribute ops evenly across Intel Xe3 GPU cores
    unsigned long long ops_per_core = ops / xe_cores_;
    unsigned long long matrix_ops_per_core = matrix_ops / xe_cores_;
    
    for (auto& core : gpu_cores_) {
        // Vector Engine throughput = 128, Matrix Engine (XMX) throughput = 1024
        unsigned long long instructions = (ops_per_core / 128) + (matrix_ops_per_core / 1024);
        // Map GPU memory access proportional to operation footprint
        unsigned long long mem_ops = (ops_per_core + matrix_ops_per_core) / 64;
        
        unsigned long long cycles = core->execute(instructions, mem_ops);
        max_cycles = std::max(max_cycles, cycles);
    }
    
    return max_cycles;
}

double Xe3GPU::get_power() const {
    double total = 0.0;
    for (const auto& core : gpu_cores_) {
        total += core->get_power();
    }
    return total;
}

double Xe3GPU::get_gflops() const {
    double total_flops = 0.0;
    for (const auto& core : gpu_cores_) {
        total_flops += (core->get_vector_flops_per_cycle() + core->get_matrix_flops_per_cycle()) * core->get_frequency_ghz();
    }
    return total_flops;
}

} // namespace panther_lake
