#pragma once

namespace panther_lake {

class Xe3GPU {
public:
    Xe3GPU(int xe_cores = 12, double frequency_ghz = 2.5);
    unsigned long long execute(unsigned long long ops, unsigned long long matrix_ops);
    double get_power() const;
    double get_gflops() const;
    int get_xe_cores() const { return xe_cores_; }

private:
    int xe_cores_;
    double frequency_ghz_;
    unsigned long long ops_completed_ = 0;
    unsigned long long cycles_elapsed_ = 0;
    double active_power_watts_ = 0.0;
    double leakage_power_watts_ = 0.0;
    
    int vector_engines_;
    int matrix_engines_;
    int ve_flops_per_cycle_ = 16;
    int xmx_flops_per_cycle_ = 128;
};

} // namespace panther_lake
