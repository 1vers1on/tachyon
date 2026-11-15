// Tachyon - low-latency spin/futex thread pool
#pragma once
#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <immintrin.h>

class ThreadPool {
public:
    explicit ThreadPool(int n) : n_(n < 1 ? 1 : n) {
        for (int i = 1; i < n_; i++) th_.emplace_back([this, i] { worker(i); });
    }
    ~ThreadPool() {
        stop_ = true;
        gen_.fetch_add(1, std::memory_order_release);
        gen_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return n_; }
    // fn(thread_id, n_threads)
    void run(const std::function<void(int, int)>& fn) {
        if (n_ == 1) { fn(0, 1); return; }
        job_ = &fn;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        gen_.notify_all();
        fn(0, n_);
        while (done_.load(std::memory_order_acquire) != n_ - 1) _mm_pause();
    }
    // split [0,n) into contiguous chunks; fn(begin, end)
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn, int64_t min_chunk = 1) {
        int nt = n_;
        if (n < min_chunk * 2 || nt == 1) { fn(0, n); return; }
        int64_t maxt = (n + min_chunk - 1) / min_chunk;
        if (nt > maxt) nt = (int)maxt;
        run([&](int tid, int) {
            if (tid >= nt) return;
            int64_t a = n * tid / nt, b = n * (tid + 1) / nt;
            if (a < b) fn(a, b);
        });
    }
private:
    void worker(int tid) {
        uint32_t last = 0;
        while (true) {
            int spins = 0;
            uint32_t g;
            while ((g = gen_.load(std::memory_order_acquire)) == last) {
                if (++spins < 200000) _mm_pause();
                else gen_.wait(last, std::memory_order_acquire);
            }
            last = g;
            if (stop_) return;
            (*job_)(tid, n_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::atomic<uint32_t> gen_{0};
    std::atomic<int> done_{0};
    const std::function<void(int, int)>* job_ = nullptr;
    volatile bool stop_ = false;
};
