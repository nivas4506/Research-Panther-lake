#pragma once
#include <vector>
#include <memory>
#include "panther_lake/cores.hpp"

namespace panther_lake {

class Xe3GPU {
public:
    Xe3GPU(int xe_cores = 12, double frequency_ghz = 1.6);
    unsigned long long execute(unsigned long long ops, unsigned long long matrix_ops);
    double get_power() const;
    double get_gflops() const;
    int get_xe_cores() const { return xe_cores_; }
    std::vector<std::unique_ptr<BlackwellGpuCore>>& get_gpu_cores() { return gpu_cores_; }
    const std::vector<std::unique_ptr<BlackwellGpuCore>>& get_gpu_cores() const { return gpu_cores_; }

private:
    int xe_cores_;
    double frequency_ghz_;
    std::vector<std::unique_ptr<BlackwellGpuCore>> gpu_cores_;
};

} // namespace panther_lake
