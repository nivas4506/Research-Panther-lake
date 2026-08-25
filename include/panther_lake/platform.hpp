#pragma once
#include <string>

namespace panther_lake {

class PlatformControllerTile {
public:
    PlatformControllerTile();
    int simulate_io(unsigned long long pcie_traffic_bytes, unsigned long long tb_traffic_bytes);
    double calculate_dvfs_voltage(double freq_ghz) const;
    double get_power() const;

private:
    unsigned long long pcie_traffic_ = 0;
    unsigned long long tb_traffic_ = 0;
    double active_power_watts_ = 0.5;
    double leakage_power_watts_ = 0.3;
    
    double pcie_max_gbs_ = 63.0;
    double tb_max_gbs_ = 10.0;
};

} // namespace panther_lake
