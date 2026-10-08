#include "host/frame_stats.h"

#include "hle/modules.h"
#include "host/gpu.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#include <unistd.h>
#else
#include "core/portable.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct ThreadTime {
    std::uint64_t ticks = 0;
    std::string name;
};

// utime + stime of every thread of this process, in clock ticks, by tid.
std::unordered_map<int, ThreadTime> thread_times() {
    std::unordered_map<int, ThreadTime> out;
#if !defined(_WIN32)
    DIR* d = opendir("/proc/self/task");
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        const int tid = std::atoi(e->d_name);
        if (tid <= 0) continue;
        char path[64];
        std::snprintf(path, sizeof(path), "/proc/self/task/%d/stat", tid);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        char buf[1024];
        const std::size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        buf[n] = 0;
        // pid (comm) state ... : comm may hold spaces, so parse after the last ')'.
        const char* open = std::strchr(buf, '(');
        const char* close = std::strrchr(buf, ')');
        if (!open || !close || close < open) continue;
        ThreadTime t;
        t.name.assign(open + 1, close);
        unsigned long long utime = 0, stime = 0;
        // Fields after ')': state(3) ppid pgrp session tty tpgid flags minflt
        // cminflt majflt cmajflt utime(14) stime(15).
        if (std::sscanf(close + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &utime, &stime) != 2) continue;
        t.ticks = utime + stime;
        out.emplace(tid, std::move(t));
    }
    closedir(d);
#else
    // Kernel + user time of the threads host_thread_set_name named, in
    // 100 ns units (core/portable.h): the same list /proc gives on Linux.
    for (const HostThreadTime& h : host_thread_times()) {
        ThreadTime t;
        t.ticks = h.ticks;
        t.name = h.name;
        out.emplace(static_cast<int>(h.tid), std::move(t));
    }
#endif
    return out;
}

struct State {
    std::mutex mu;
    Clock::time_point window_start{}, last_flip{};
    std::uint64_t displayed_at_start = 0;  // completed flips when the window opened
    std::vector<double> intervals_ms;
    std::unordered_map<int, ThreadTime> threads;
    GpuBusy gpu{};  // host_gpu_busy() when the window opened
};

}  // namespace

void frame_stats_on_flip() {
    const bool on = frame_stats_enabled();
    if (!on) return;
    static State s;
    static const long hz = [] {
#if !defined(_WIN32)
        return sysconf(_SC_CLK_TCK);
#else
        return 10000000L;  // GetThreadTimes counts 100 ns units
#endif
    }();
    std::lock_guard<std::mutex> lk(s.mu);
    const auto now = Clock::now();
    if (s.last_flip.time_since_epoch().count() == 0) {
        s.window_start = s.last_flip = now;
        s.threads = thread_times();
        s.gpu = host_gpu_busy();
        return;
    }
    s.intervals_ms.push_back(std::chrono::duration<double, std::milli>(now - s.last_flip).count());
    s.last_flip = now;
    const double secs = std::chrono::duration<double>(now - s.window_start).count();
    if (secs < 1.0) return;

    double sum = 0, worst = 0;
    int over16 = 0, over33 = 0;
    for (const double v : s.intervals_ms) {
        sum += v;
        worst = std::max(worst, v);
        over16 += v > 17.5;
        over33 += v > 34.5;
    }
    const std::size_t n = s.intervals_ms.size();
    std::vector<double> sorted = s.intervals_ms;
    std::sort(sorted.begin(), sorted.end());
    const double p95 = sorted[std::min(n - 1, static_cast<std::size_t>(0.95 * static_cast<double>(n)))];

    auto cur = thread_times();
    struct Busy {
        double pct;
        std::string name;
    };
    std::vector<Busy> busy;
    double total = 0;
    for (const auto& [tid, t] : cur) {
        const auto it = s.threads.find(tid);
        const std::uint64_t was = it != s.threads.end() ? it->second.ticks : 0;
        const double pct = 100.0 * static_cast<double>(t.ticks - was) / static_cast<double>(hz) / secs;
        total += pct;
        if (pct >= 5.0) busy.push_back({pct, t.name + ":" + std::to_string(tid)});
    }
    std::sort(busy.begin(), busy.end(), [](const Busy& a, const Busy& b) { return a.pct > b.pct; });
    std::string top;
    for (std::size_t i = 0; i < busy.size() && i < 8; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " %s %.0f%%", busy[i].name.c_str(), busy[i].pct);
        top += buf;
    }
    const std::uint64_t sync_ns = main_thread_waited_ns(0), sleep_ns = main_thread_waited_ns(1), file_ns = main_thread_waited_ns(2);
    std::string work;
    if (std::vector<std::uint32_t> busy = main_frames_busy_us(); !busy.empty()) {
        std::sort(busy.begin(), busy.end());
        double busy_sum = 0;
        int busy_over = 0;
        for (std::uint32_t us : busy) {
            busy_sum += us;
            busy_over += us > 16667;
        }
        char wb[160];
        std::snprintf(wb, sizeof(wb), "; main loop work avg %.1f p95 %.1f max %.1f ms, %d over 16.7", busy_sum / busy.size() / 1e3,
                      busy[busy.size() * 95 / 100] / 1e3, busy.back() / 1e3, busy_over);
        work = wb;
    }
    // The flips above are the game's submits. What the display actually showed
    // is the completion counter, which the flip clock paces: the
    // two differ when the game submits faster than the display can show.
    const std::uint64_t displayed_now = hle_video_flip_count();
    const double displayed = static_cast<double>(displayed_now - s.displayed_at_start) / secs;
    s.displayed_at_start = displayed_now;
    // How much of the second our work kept the GPU busy (host/gpu_busy.cpp:
    // the union of our command buffers' spans on the GPU's clock), last on
    // the line so the summaries' patterns for what is before it still hold.
    // The spans are counted a frame or two after they ran.
    std::string gpu;
    const GpuBusy gpu_now = host_gpu_busy();
    if (gpu_now.on && s.gpu.on) {
        const double ms = static_cast<double>(gpu_now.busy_ns - s.gpu.busy_ns) / 1e6 / secs;
        char gb[96];
        std::snprintf(gb, sizeof(gb), "; gpu busy %.0f ms/s (%.0f%%)", ms, ms / 10.0);
        gpu = gb;
    }
    s.gpu = gpu_now;
    host_log("frames: %.2f s: %zu flips (%.1f/s), displayed %.1f/s, interval avg %.1f p95 %.1f max %.1f ms, %d over 16.7, %d over 33.3; "
             "cpu %.0f%%:%s; main loop waited %.0f ms, slept %.0f ms, file I/O %.0f ms%s%s",
             secs, n, static_cast<double>(n) / secs, displayed, sum / static_cast<double>(n), p95, worst, over16, over33, total,
             top.c_str(), static_cast<double>(sync_ns) / 1e6, static_cast<double>(sleep_ns) / 1e6,
             static_cast<double>(file_ns) / 1e6, work.c_str(), gpu.c_str());
    s.intervals_ms.clear();
    s.window_start = now;
    s.threads = std::move(cur);
}
