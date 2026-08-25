#pragma once
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace sycl {

enum class device_type {
    cpu,
    gpu,
    npu
};

class device {
public:
    device(device_type t) : type_(t) {}
    device_type get_type() const { return type_; }
    std::string get_name() const {
        switch(type_) {
            case device_type::cpu: return "Cougar Cove / Darkmont Hybrid CPU";
            case device_type::gpu: return "Xe3-LPG Celestial GPU";
            case device_type::npu: return "Intel NPU 5 Accelerator";
        }
        return "Unknown Device";
    }
private:
    device_type type_;
};

class handler {
public:
    template <typename KernelName, typename Func>
    void parallel_for(size_t range, Func f) {
        // Mock kernel registration
        registered_kernels_.push_back([f, range]() {
            for (size_t i = 0; i < range; ++i) {
                f(i);
            }
        });
    }
    
    void run_all() {
        for (auto& k : registered_kernels_) {
            k();
        }
    }
private:
    std::vector<std::function<void()>> registered_kernels_;
};

class queue {
public:
    queue(device dev) : dev_(dev) {}
    device get_device() const { return dev_; }
    
    template <typename Func>
    void submit(Func f) {
        handler h;
        f(h);
        h.run_all();
    }
private:
    device dev_;
};

template <typename T>
class buffer {
public:
    buffer(T* data, size_t count) : data_(data), count_(count) {}
    size_t get_count() const { return count_; }
private:
    T* data_;
    size_t count_;
};

enum class access_mode {
    read,
    write,
    read_write
};

template <typename T, access_mode Mode = access_mode::read_write>
class accessor {
public:
    accessor(buffer<T>& buf, handler& h) : count_(buf.get_count()) {}
    size_t size() const { return count_; }
private:
    size_t count_;
};

} // namespace sycl
