#include "panther_lake/cores.hpp"
#include <cmath>

namespace panther_lake {

ExecutionCore::ExecutionCore(int core_id, std::string type, double frequency_ghz, double ipc_target)
    : core_id_(core_id), type_(type), frequency_ghz_(frequency_ghz), ipc_target_(ipc_target) {}

double ExecutionCore::get_power() const {
    return active_power_watts_ + leakage_power_watts_;
}

CougarCoveCore::CougarCoveCore(int core_id, double frequency_ghz)
    : CpuCore(core_id, "P-Core (Cougar Cove)", frequency_ghz, 3.0) {
    leakage_power_watts_ = 0.5;
}

unsigned long long CougarCoveCore::execute(unsigned long long instructions, unsigned long long mem_ops) {
    unsigned long long hits = static_cast<unsigned long long>(mem_ops * 0.95);
    unsigned long long misses = mem_ops - hits;
    l1_hits_ += hits;
    l1_misses_ += misses;
    
    double exec_cycles = instructions / ipc_target_;
    double mem_penalty = (hits * 4) + (misses * 15);
    unsigned long long total_cycles = static_cast<unsigned long long>(exec_cycles + (mem_penalty * 0.4));
    
    cycles_elapsed_ += total_cycles;
    instructions_retired_ += instructions;
    active_power_watts_ = 4.0 * std::pow(frequency_ghz_ / 5.0, 3);
    
    return total_cycles;
}

DarkmontCore::DarkmontCore(int core_id, bool is_lp, double frequency_ghz)
    : CpuCore(core_id, is_lp ? "LP E-Core" : "E-Core (Darkmont)", is_lp ? 2.5 : frequency_ghz, is_lp ? 1.1 : 1.5), is_lp_(is_lp) {
    leakage_power_watts_ = is_lp ? 0.05 : 0.1;
}

unsigned long long DarkmontCore::execute(unsigned long long instructions, unsigned long long mem_ops) {
    unsigned long long hits = static_cast<unsigned long long>(mem_ops * 0.9);
    unsigned long long misses = mem_ops - hits;
    l1_hits_ += hits;
    l1_misses_ += misses;
    
    double exec_cycles = instructions / ipc_target_;
    double mem_penalty = (hits * 4) + (misses * 20);
    unsigned long long total_cycles = static_cast<unsigned long long>(exec_cycles + (mem_penalty * 0.8));
    
    cycles_elapsed_ += total_cycles;
    instructions_retired_ += instructions;
    
    double base_power = is_lp_ ? 0.5 : 1.2;
    active_power_watts_ = base_power * std::pow(frequency_ghz_ / 3.0, 3);
    
    return total_cycles;
}

Xe3GpuCore::Xe3GpuCore(int core_id, double frequency_ghz)
    : ExecutionCore(core_id, "Intel Xe3 GPU Core", frequency_ghz, 32.0) {
    leakage_power_watts_ = 0.10;
}

unsigned long long Xe3GpuCore::execute(unsigned long long instructions, unsigned long long mem_ops) {
    unsigned long long hits = static_cast<unsigned long long>(mem_ops * 0.98);
    unsigned long long misses = mem_ops - hits;
    l1_hits_ += hits;
    l1_misses_ += misses;
    
    double exec_cycles = instructions / ipc_target_;
    double mem_penalty = (hits * 2) + (misses * 20); // GPU has wide memory paths
    unsigned long long total_cycles = static_cast<unsigned long long>(exec_cycles + (mem_penalty * 0.2));
    
    cycles_elapsed_ += total_cycles;
    instructions_retired_ += instructions;
    
    // Intel Xe3 core dynamic power: scales with frequency cubed
    active_power_watts_ = 1.8 * std::pow(frequency_ghz_ / 1.6, 3);
    
    return total_cycles;
}

} // namespace panther_lake
