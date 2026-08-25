#pragma once
#include <string>

namespace panther_lake {

class FoverosInterconnect {
public:
    FoverosInterconnect();
    unsigned long long route(const std::string& src, const std::string& dst, unsigned long long data_size_bytes);
    
    unsigned long long get_transfers_count() const { return transfers_count_; }
    unsigned long long get_total_bytes_routed() const { return total_bytes_routed_; }
    unsigned long long get_contention_count() const { return contention_count_; }

private:
    unsigned long long transfers_count_ = 0;
    unsigned long long total_bytes_routed_ = 0;
    unsigned long long contention_count_ = 0;
};

} // namespace panther_lake
