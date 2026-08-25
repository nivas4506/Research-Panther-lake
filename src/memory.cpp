#include "panther_lake/memory.hpp"
#include <cstdlib>

namespace panther_lake {

LPDDR5XController::LPDDR5XController(double speed_mts, int channels)
    : speed_mts_(speed_mts), channels_(channels) {}

double LPDDR5XController::get_peak_bandwidth_gbs() const {
    double width_bytes = data_width_bits_ / 8.0;
    return (speed_mts_ * 1000000.0 * width_bytes * channels_) / 1000000000.0;
}

double LPDDR5XController::access_latency_ns(bool is_write) const {
    double base = is_write ? 35.0 : 45.0;
    double queue_delay = 5.0 + (std::rand() % 10);
    return base + queue_delay;
}

void LPDDR5XController::transfer(size_t size_bytes, bool is_write) {
    if (is_write) {
        writes_++;
    } else {
        reads_++;
    }
    bytes_transferred_ += size_bytes;
}

SystemLevelCache::SystemLevelCache(int size_mb) : size_mb_(size_mb) {}

std::pair<bool, int> SystemLevelCache::access(unsigned long long address) {
    unsigned long long line = address >> 6;
    if (cached_lines_.count(line)) {
        hits_++;
        return {true, latency_cycles_};
    } else {
        misses_++;
        if (cached_lines_.size() > (size_mb_ * 1024ULL * 1024ULL / 64ULL)) {
            // Evict the first cache line as a simple heuristic
            cached_lines_.erase(cached_lines_.begin());
        }
        cached_lines_.insert(line);
        return {false, latency_cycles_};
    }
}

} // namespace panther_lake
