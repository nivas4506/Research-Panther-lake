#include "panther_lake/npu.hpp"
#include <algorithm>

namespace panther_lake {

NPU5::NPU5(double frequency_ghz) : frequency_ghz_(frequency_ghz) {
    reset_active_power(true);
}

void NPU5::reset_active_power(bool active) {
    if (active) {
        active_power_watts_ = 8.0 * (frequency_ghz_ / 1.8);
    } else {
        active_power_watts_ = 0.0;
    }
}

unsigned long long NPU5::execute_matmul(int m, int n, int k) {
    unsigned long long total_macs = static_cast<unsigned long long>(m) * n * k;
    double capacity_per_cycle = num_nce_ * macs_per_nce_cycle_;
    
    double compute_cycles = total_macs / capacity_per_cycle;
    unsigned long long total_cycles = static_cast<unsigned long long>(compute_cycles * 1.1);
    
    if (total_cycles == 0 && total_macs > 0) {
        total_cycles = 1;
    }
    
    cycles_elapsed_ += total_cycles;
    macs_completed_ += total_macs;
    
    double utilization = std::min(1.0, compute_cycles / std::max(1ULL, total_cycles));
    active_power_watts_ = 8.0 * (frequency_ghz_ / 1.8) * utilization;
    
    return total_cycles;
}

double NPU5::get_power() const {
    return active_power_watts_ + leakage_power_watts_;
}

double NPU5::get_max_tops() const {
    double ops_per_cycle = num_nce_ * macs_per_nce_cycle_ * 2;
    return (ops_per_cycle * frequency_ghz_) / 1000.0;
}

} // namespace panther_lake
