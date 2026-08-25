#include "panther_lake/gpu.hpp"
#include <cmath>
#include <algorithm>

namespace panther_lake {

Xe3GPU::Xe3GPU(int xe_cores, double frequency_ghz)
    : xe_cores_(xe_cores), frequency_ghz_(frequency_ghz) {
    leakage_power_watts_ = 0.25 * xe_cores_;
    vector_engines_ = xe_cores_ * 8;
    matrix_engines_ = xe_cores_ * 8;
}

unsigned long long Xe3GPU::execute(unsigned long long ops, unsigned long long matrix_ops) {
    double vector_capacity = vector_engines_ * ve_flops_per_cycle_;
    double matrix_capacity = matrix_engines_ * xmx_flops_per_cycle_;
    
    double ve_cycles = vector_capacity > 0 ? ops / vector_capacity : 0;
    double xmx_cycles = matrix_capacity > 0 ? matrix_ops / matrix_capacity : 0;
    
    unsigned long long total_cycles = static_cast<unsigned long long>(
        std::max(ve_cycles, xmx_cycles) + 0.8 * std::min(ve_cycles, xmx_cycles)
    );
    
    if (total_cycles == 0 && (ops > 0 || matrix_ops > 0)) {
        total_cycles = 1;
    }
    
    cycles_elapsed_ += total_cycles;
    ops_completed_ += (ops + matrix_ops);
    
    double dynamic_power_factor = 2.5 * (static_cast<double>(xe_cores_) / 12.0);
    active_power_watts_ = dynamic_power_factor * std::pow(frequency_ghz_ / 2.5, 3);
    
    return total_cycles;
}

double Xe3GPU::get_power() const {
    return active_power_watts_ + leakage_power_watts_;
}

double Xe3GPU::get_gflops() const {
    double ve_gflops = vector_engines_ * ve_flops_per_cycle_ * frequency_ghz_;
    double xmx_gflops = matrix_engines_ * xmx_flops_per_cycle_ * frequency_ghz_;
    return ve_gflops + xmx_gflops;
}

} // namespace panther_lake
