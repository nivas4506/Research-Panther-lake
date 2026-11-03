#pragma once
#include <cstddef>
#include <unordered_set>
#include <utility>

namespace panther_lake {

class LPDDR5XController {
public:
    LPDDR5XController(double speed_mts = 9600.0, int channels = 2);
    double get_peak_bandwidth_gbs() const;
    double access_latency_ns(bool is_write) const;
    void transfer(size_t size_bytes, bool is_write);
    
    unsigned long long get_reads() const { return reads_; }
    unsigned long long get_writes() const { return writes_; }
    unsigned long long get_bytes_transferred() const { return bytes_transferred_; }

private:
    double speed_mts_;
    int channels_;
    int data_width_bits_ = 64;
    unsigned long long reads_ = 0;
    unsigned long long writes_ = 0;
    unsigned long long bytes_transferred_ = 0;
};

class SystemLevelCache {
public:
    SystemLevelCache(int size_mb = 24);
    std::pair<bool, int> access(unsigned long long address);
    
    unsigned long long get_hits() const { return hits_; }
    unsigned long long get_misses() const { return misses_; }
    int get_size_mb() const { return size_mb_; }

private:
    int size_mb_;
    int latency_cycles_ = 45;
    unsigned long long hits_ = 0;
    unsigned long long misses_ = 0;
    std::unordered_set<unsigned long long> cached_lines_;
};

} // namespace panther_lake
