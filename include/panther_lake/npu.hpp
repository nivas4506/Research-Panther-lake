#pragma once

namespace panther_lake {

class NPU5 {
public:
    NPU5(double frequency_ghz = 1.8);
    unsigned long long execute_matmul(int m, int n, int k);
    double get_power() const;
    double get_max_tops() const;
    unsigned long long get_macs_completed() const { return macs_completed_; }

private:
    double frequency_ghz_;
    unsigned long long macs_completed_ = 0;
    unsigned long long cycles_elapsed_ = 0;
    double active_power_watts_ = 0.0;
    double leakage_power_watts_ = 0.2;
    
    int num_nce_ = 4;
    int macs_per_nce_cycle_ = 4096;
};

} // namespace panther_lake
