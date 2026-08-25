#pragma once
#include <string>

namespace panther_lake {

class ExecutionCore {
public:
    ExecutionCore(int core_id, std::string type, double frequency_ghz, double ipc_target);
    virtual ~ExecutionCore() = default;
    
    virtual unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) = 0;
    double get_power() const;
    double get_frequency_ghz() const { return frequency_ghz_; }
    void set_frequency_ghz(double freq) { frequency_ghz_ = freq; }
    unsigned long long get_retired_instructions() const { return instructions_retired_; }
    unsigned long long get_cycles_elapsed() const { return cycles_elapsed_; }
    unsigned long long get_l1_hits() const { return l1_hits_; }
    unsigned long long get_l1_misses() const { return l1_misses_; }

protected:
    int core_id_;
    std::string type_;
    double frequency_ghz_;
    double ipc_target_;
    unsigned long long instructions_retired_ = 0;
    unsigned long long cycles_elapsed_ = 0;
    double active_power_watts_ = 0.0;
    double leakage_power_watts_ = 0.0;
    unsigned long long l1_hits_ = 0;
    unsigned long long l1_misses_ = 0;
};

using CpuCore = ExecutionCore; // For compatibility

class CougarCoveCore : public ExecutionCore {
public:
    CougarCoveCore(int core_id, double frequency_ghz = 5.0);
    unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) override;
};

class DarkmontCore : public ExecutionCore {
public:
    DarkmontCore(int core_id, bool is_lp, double frequency_ghz = 3.5);
    unsigned long long execute(unsigned long long instructions, unsigned long long mem_ops) override;
private:
    bool is_lp_;
};

} // namespace panther_lake
