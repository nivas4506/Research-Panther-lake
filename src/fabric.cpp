#include "panther_lake/fabric.hpp"
#include <cmath>

namespace panther_lake {

FoverosInterconnect::FoverosInterconnect() {}

int FoverosInterconnect::route(const std::string& src, const std::string& dst, unsigned long long data_size_bytes) {
    transfers_count_++;
    total_bytes_routed_ += data_size_bytes;
    
    double bandwidth = 128.0; // GB/s base
    int base_lat = 12;
    
    if ((src == "ComputeTile" && dst == "GraphicsTile") || (src == "GraphicsTile" && dst == "ComputeTile")) {
        bandwidth = 256.0;
        base_lat = 8;
    } else if ((src == "GraphicsTile" && dst == "PlatformControllerTile") || (src == "PlatformControllerTile" && dst == "GraphicsTile")) {
        bandwidth = 64.0;
        base_lat = 16;
    }
    
    double transfer_ns = (data_size_bytes / (bandwidth * 1000000000.0)) * 1000000000.0;
    int transfer_cycles = static_cast<int>(transfer_ns / 0.5); // 2.0 GHz base fabric clock
    
    int contention_cycles = 0;
    if (transfers_count_ % 100 == 0) {
        contention_count_++;
        contention_cycles = 5;
    }
    
    return base_lat + transfer_cycles + contention_cycles;
}

} // namespace panther_lake
