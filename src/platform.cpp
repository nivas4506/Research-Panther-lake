#include "panther_lake/platform.hpp"
#include <algorithm>

namespace panther_lake {

PlatformControllerTile::PlatformControllerTile() {}

int PlatformControllerTile::simulate_io(unsigned long long pcie_traffic_bytes, unsigned long long tb_traffic_bytes) {
    pcie_traffic_ += pcie_traffic_bytes;
    tb_traffic_ += tb_traffic_bytes;
    
    double pcie_transfer_ns = (pcie_traffic_bytes / (pcie_max_gbs_ * 1000000000.0)) * 1000000000.0;
    double tb_transfer_ns = (tb_traffic_bytes / (tb_max_gbs_ * 1000000000.0)) * 1000000000.0;
    
    int pct_cycles = static_cast<int>(std::max(pcie_transfer_ns, tb_transfer_ns));
    
    double total_traffic = pcie_traffic_bytes + tb_traffic_bytes;
    double io_utilization = std::min(1.0, total_traffic / 10000000.0);
    active_power_watts_ = 0.5 + 2.0 * io_utilization;
    
    return std::max(1, pct_cycles);
}

double PlatformControllerTile::calculate_dvfs_voltage(double freq_ghz) const {
    if (freq_ghz <= 1.0) return 0.70;
    if (freq_ghz >= 6.0) return 1.35;
    double ratio = (freq_ghz - 1.0) / 5.0;
    return 0.70 + ratio * (1.35 - 0.70);
}

double PlatformControllerTile::get_power() const {
    return active_power_watts_ + leakage_power_watts_;
}

void PlatformControllerTile::reset_active_power() {
    active_power_watts_ = 0.5;
}

} // namespace panther_lake
