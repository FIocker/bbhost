// Draw recording on its own thread (BBHOST_RECORDER, on unless 0).
//
// Of each draw's ~11 us on the command processor, ~2.5 went to the driver:
// vkUpdateDescriptorSets, then the binds, dynamic state and draw call. A draw
// now puts those in a packet, and a recorder thread replays the packets into
// the command buffer in order while the command processor resolves the next
// draw. Everything else that records - passes, transfers, copies, clears,
// dispatches, submission - takes the command buffer through g_cmd(), which
// waits for the recorder to finish the packets published so far. So there is
// one ordered stream of commands, whichever thread records them, and the
// recorder only ever runs while the command processor (holding g.mu) keeps
// publishing draws.
//
// Only the thread holding g.mu publishes or drains. The recorder takes no
// lock: it owns the command buffer between a publish and the drain that
// follows. It records nothing but draw packets, and the descriptor sets it
// writes belong to the packet (vkUpdateDescriptorSets needs only the set
// synchronised, not its pool, so the command processor keeps allocating).
// The buffer shadow's copies go to a command buffer of their own pool.

#include "host/gpu_internal.h"

#include <algorithm>
#include "core/portable.h"

#include "log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace gpu {

namespace {

inline void cpu_relax() {
#if defined(__x86_64__)
    _mm_pause();
#endif
}

// Flushes this core's write-combining buffers (see recorder_thread).
inline void write_combine_fence() {
#if defined(__x86_64__)
    _mm_sfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

struct DrawPacket {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    // Descriptor writes, applied first and in order. Each write's info
    // pointer is an index into buffers / images until the replay.
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkDescriptorImageInfo> images;
    std::vector<std::int32_t> write_buffer, write_image;  // first info of each write, or -1
    // take_sets: a draw's writes, their pointers already final.
    std::vector<VkWriteDescriptorSet> taken_writes;
    std::vector<VkDescriptorImageInfo> taken_images;
    std::vector<VkDescriptorBufferInfo> taken_buffers;
    VkDescriptorBufferInfo extra[2] = {};

    VkPipeline pipeline = VK_NULL_HANDLE;  // null: keep the bound one
    bool library = false;
    DrawLibraryState lib{};
    bool has_depth_bounds = false;
    bool bind_sets = false;
    bool bindless = false;  // set 3, the global views and samplers, too
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet sets[2] = {};
    std::uint32_t dynamic_count = 0, dynamic[2] = {};
    std::uint32_t pushes = 0;  // constant buffers pushed into set 2 after the binds: binding and range
    std::uint32_t push_binding[2 * gcn::kMaxBuffers];
    VkDescriptorBufferInfo push_infos[2 * gcn::kMaxBuffers];
    bool viewport = false, scissor = false, bounds = false, stencil = false, blend = false;
    VkViewport vp{};
    VkRect2D sc{};
    float depth_bounds[2] = {};
    std::uint32_t stencil_words[6] = {};
    float blend_const[4] = {};
    bool index = false;
    VkBuffer index_buffer = VK_NULL_HANDLE;
    VkDeviceSize index_offset = 0;
    VkIndexType index_type = VK_INDEX_TYPE_UINT16;
    std::uint32_t vertex_buffers = 0;
    VkBuffer vb[kMaxVertexBindings] = {};
    VkDeviceSize vb_offset[kMaxVertexBindings] = {};
    DrawCall call{};
    bool draw = false;
    // Pass changes, replayed after the descriptor writes and before the binds.
    bool end_pass = false, begin_pass = false;
    bool end_pass_barrier = true;  // the end's barrier with it (lazy pass barriers leave it out)
    bool pass_barrier = false;     // a pass end's barrier owed from earlier, after end_pass
    // A transfer batch, between the pass end and the pass begin: the barrier
    // into the transfer stage, buffer copies, the barrier out of it.
    bool xfer_begin = false, xfer_end = false;
    // A copy-version batch is ~25 copies (G9); past this many the rest were
    // recorded in place, after waiting for the recorder to drain.
    static constexpr std::uint32_t kCopies = 32;
    std::uint32_t ncopies = 0;
    std::uint32_t copy_orders = 0;  // bit k: the copies before copy k finish first (copy_order)
    VkBuffer copy_src[kCopies] = {}, copy_dst[kCopies] = {};
    VkBufferCopy copies[kCopies] = {};
    // Region copies (textures.cpp rt_copied_view), after the transfer batch:
    // a draw binds up to a few of a glare pyramid's levels.
    static constexpr std::uint32_t kRegionCopies = 4;
    std::uint32_t nregions = 0;
    struct RegionCopy {
        VkImage src, dst;
        std::uint32_t width, height;
        bool dst_initialised;
    } regions[kRegionCopies] = {};
    VkRenderingInfo rendering{};
    VkRenderingAttachmentInfo colors[8]{}, depth_att{}, stencil_att{};
    bool has_depth = false, has_stencil = false;

    void reset() {
        taken_writes.clear();
        writes.clear();
        buffers.clear();
        images.clear();
        write_buffer.clear();
        write_image.clear();
        pipeline = VK_NULL_HANDLE;
        library = bind_sets = viewport = scissor = bounds = stencil = blend = index = draw = false;
        end_pass = begin_pass = false;
        end_pass_barrier = true;
        pass_barrier = false;
        xfer_begin = xfer_end = false;
        ncopies = 0;
        copy_orders = 0;
        nregions = 0;
        pushes = 0;
        vertex_buffers = 0;
    }
};

// Nothing a pass change must precede is in the packet yet.
bool before_binds(const DrawPacket& p) {
    return !p.pipeline && !p.library && !p.bind_sets && !p.pushes && !p.viewport && !p.scissor && !p.bounds && !p.stencil && !p.blend &&
           !p.index && !p.vertex_buffers && !p.draw;
}

constexpr std::uint64_t kRing = 256;
DrawPacket* g_ring = nullptr;
std::atomic<std::uint64_t> g_published{0};  // packets handed over (written under g.mu)
std::atomic<std::uint64_t> g_done{0};       // packets replayed (written by the recorder)
std::atomic<bool> g_sleeping{false};
std::uint64_t g_drains = 0, g_drain_waits = 0, g_full_waits = 0;
std::atomic<std::uint64_t> g_replayed{0};
DrawCmds* g_open = nullptr;  // under g.mu

const bool g_recorder_on = [] {
    const char* e = std::getenv("BBHOST_RECORDER");
    return !(e && e[0] == '0');
}();
// BBHOST_RECORDER_INLINE=1 (a diagnostic): the packets are replayed on the
// publishing thread at once, no recorder thread - the packet path's command
// order, recorded in place.
const bool g_recorder_inline = [] {
    const char* e = std::getenv("BBHOST_RECORDER_INLINE");
    return e && e[0] == '1';
}();

void replay(DrawPacket& p) {
    const VkCommandBuffer cmd = p.cmd;
    if (!p.taken_writes.empty()) {
        vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(p.taken_writes.size()), p.taken_writes.data(), 0, nullptr);
    }
    if (!p.writes.empty()) {
        for (std::size_t i = 0; i < p.writes.size(); ++i) {
            VkWriteDescriptorSet& w = p.writes[i];
            if (p.write_buffer[i] >= 0) w.pBufferInfo = &p.buffers[static_cast<std::size_t>(p.write_buffer[i])];
            if (p.write_image[i] >= 0) w.pImageInfo = &p.images[static_cast<std::size_t>(p.write_image[i])];
        }
        vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(p.writes.size()), p.writes.data(), 0, nullptr);
    }
    if (p.end_pass) record_end_rendering(cmd, p.end_pass_barrier);
    if (p.pass_barrier) record_pass_barrier(cmd);
    if (p.xfer_begin) record_transfer_barrier(cmd, true);
    for (std::uint32_t k = 0; k < p.ncopies; ++k) {
        if (p.copy_orders >> k & 1) record_copy_order_barrier(cmd);
        vkCmdCopyBuffer(cmd, p.copy_src[k], p.copy_dst[k], 1, &p.copies[k]);
    }
    if (p.xfer_end) record_transfer_barrier(cmd, false);
    for (std::uint32_t k = 0; k < p.nregions; ++k) {
        const DrawPacket::RegionCopy& r = p.regions[k];
        record_region_copy(cmd, r.src, r.dst, r.width, r.height, r.dst_initialised);
    }
    if (p.begin_pass) {
        p.rendering.pColorAttachments = p.colors;
        p.rendering.pDepthAttachment = p.has_depth ? &p.depth_att : nullptr;
        p.rendering.pStencilAttachment = p.has_stencil ? &p.stencil_att : nullptr;
        vkCmdBeginRendering(cmd, &p.rendering);
    }
    if (p.pipeline) vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
    if (p.library) record_library_state(cmd, p.lib, p.has_depth_bounds);
    if (p.bind_sets) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 2, p.sets, p.dynamic_count, p.dynamic);
    if (p.bind_sets && p.bindless) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 3, 1, &g.bindless_set, 0, nullptr);
    if (p.pushes) {
        VkWriteDescriptorSet writes[2 * gcn::kMaxBuffers];
        for (std::uint32_t k = 0; k < p.pushes; ++k) {
            writes[k] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[k].dstBinding = p.push_binding[k];
            writes[k].descriptorCount = 1;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[k].pBufferInfo = &p.push_infos[k];
        }
        g.cmd_push_descriptor_set(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 2, p.pushes, writes);
    }
    if (p.viewport) vkCmdSetViewport(cmd, 0, 1, &p.vp);
    if (p.scissor) vkCmdSetScissor(cmd, 0, 1, &p.sc);
    if (p.bounds) vkCmdSetDepthBounds(cmd, p.depth_bounds[0], p.depth_bounds[1]);
    if (p.stencil) record_stencil_words(cmd, p.stencil_words);
    if (p.blend) vkCmdSetBlendConstants(cmd, p.blend_const);
    if (p.index) vkCmdBindIndexBuffer(cmd, p.index_buffer, p.index_offset, p.index_type);
    if (p.vertex_buffers) vkCmdBindVertexBuffers(cmd, 0, p.vertex_buffers, p.vb, p.vb_offset);
    if (p.draw) record_draw_call(cmd, p.call);
}

// How long the recorder spins for the next packet before it sleeps, in
// pauses (BBHOST_RECORDER_SPIN, 4000). That rides the few microseconds between
// a burst's draws. On a Steam Deck 61% of the recorder's time is this loop and
// 12% its work, but spinning less did not give its GPU the power: 300 and 50
// pauses, windowed, 56.7 and 56.6 fps against 57.0 - the command processor
// waited longer on the recorder's wake-ups instead.
const int g_recorder_spin = [] {
    const char* e = std::getenv("BBHOST_RECORDER_SPIN");
    return e && *e ? std::max(0, std::atoi(e)) : 4000;
}();

void recorder_thread() {
    host_thread_set_name("bb-record");
    const int spin = g_recorder_spin;
    std::uint64_t done = 0;
    for (;;) {
        std::uint64_t published = g_published.load(std::memory_order_acquire);
        if (published == done) {
            // Draws come in bursts a few microseconds apart: spin through
            // those, sleep through the gaps between frames.
            for (int i = 0; i < spin && published == done; ++i) {
                cpu_relax();
                published = g_published.load(std::memory_order_acquire);
            }
            if (published == done) {
                g_sleeping.store(true, std::memory_order_seq_cst);
                g_published.wait(done, std::memory_order_seq_cst);
                g_sleeping.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        while (done < published) {
            replay(g_ring[done % kRing]);
            ++done;
            // The driver may record into write-combined memory (AMD's
            // Windows driver does: its command chunks are GPU memory the
            // CPU writes through). Write-combining buffers are per core and
            // x86's ordering does not cover them - a release store is a
            // plain store - so without a store fence the command processor
            // could end and submit this command buffer while some of the
            // packet's commands still sat in this core's buffers, and the
            // GPU ran whatever was in memory: a device loss in the title's
            // first frames on a Radeon 8060S, every run, while in-place
            // recording (the same commands, one thread) never was.
            write_combine_fence();
            g_done.store(done, std::memory_order_release);
        }
        g_replayed.store(done, std::memory_order_relaxed);
    }
}

void ensure_started() {
    static const bool started = [] {
        g_ring = new DrawPacket[kRing];  // never freed: the thread runs until exit
        if (g_recorder_inline) {
            host_log("recorder: packets replayed in place as they are published (BBHOST_RECORDER_INLINE)");
            return true;
        }
        std::thread(recorder_thread).detach();
        host_log("recorder: draws are recorded on their own thread (BBHOST_RECORDER=0 records them in place)");
        return true;
    }();
    (void)started;
}

}  // namespace

bool recorder_enabled() { return g_recorder_on; }

void recorder_drain() {
    if (!g_ring) return;
    const std::uint64_t published = g_published.load(std::memory_order_relaxed);  // only the g.mu holder publishes
    ++g_drains;
    if (g_done.load(std::memory_order_acquire) == published) return;
    ++g_drain_waits;
    for (std::uint64_t spins = 0; g_done.load(std::memory_order_acquire) != published; ++spins) {
        if ((spins & 1023) == 0 && g_sleeping.load(std::memory_order_seq_cst)) g_published.notify_one();  // see publish()
        if (spins < (1u << 16)) {
            cpu_relax();
        } else {
            std::this_thread::yield();
        }
    }
}

VkCommandBuffer g_cmd() {
    if (g.profile_gaps) profile_note_site_locked(__builtin_return_address(0));
    // An open draw whose packet already holds commands goes first. Descriptor
    // writes alone need not: they only have to land before the draw's own
    // binds, which come after them in the same packet.
    if (g_open && g_open->holds_commands()) g_open->publish();
    recorder_drain();
    return g.cmd_;
}

std::string recorder_report() {
    if (!g_ring) return std::string();
    char buf[200];
    std::snprintf(buf, sizeof(buf), "recorder: %llu draws replayed; %llu drains, %llu of them waited; %llu waits for a free packet",
                  static_cast<unsigned long long>(g_replayed.load()), static_cast<unsigned long long>(g_drains),
                  static_cast<unsigned long long>(g_drain_waits), static_cast<unsigned long long>(g_full_waits));
    return buf;
}

}  // namespace gpu

std::string host_gpu_recorder_report() {
    std::lock_guard<GpuMutex> lock(gpu::g.mu);
    std::string r = gpu::recorder_report();
    if (gpu::set_cache_on()) r += "; " + gpu::set_cache_report();
    if (const std::string cv = gpu::copy_versions_report(); !cv.empty()) r += "; " + cv;
    if (const std::string pb = gpu::render_barriers_report(); !pb.empty()) r += "; " + pb;
    return r;
}

namespace gpu {

// ---- DrawCmds: a draw's commands, packed or recorded in place ----

DrawCmds::DrawCmds(bool deferred) { start(deferred); }

void DrawCmds::start(bool deferred) {
    if (packet_) {
        if (!deferred) publish();  // started early, and the draw records in place after all
        return;
    }
    if (!deferred || !g_recorder_on) return;
    ensure_started();
    // Two open packets would share a ring slot: one another draw left open goes first.
    if (g_open && g_open != this) g_open->publish();
    const std::uint64_t published = g_published.load(std::memory_order_relaxed);
    if (published - g_done.load(std::memory_order_acquire) >= kRing) {
        ++g_full_waits;
        while (published - g_done.load(std::memory_order_acquire) >= kRing) cpu_relax();
    }
    DrawPacket& p = g_ring[published % kRing];
    p.reset();
    packet_ = &p;
    g_open = this;
}

DrawCmds* DrawCmds::open() { return g_open; }

bool DrawCmds::holds_commands() const {
    if (!packet_) return false;
    const DrawPacket& p = *static_cast<const DrawPacket*>(packet_);
    return !before_binds(p) || p.end_pass || p.pass_barrier || p.begin_pass || p.xfer_begin || p.xfer_end || p.ncopies || p.nregions;
}

DrawCmds::~DrawCmds() { publish(); }

void DrawCmds::publish() {
    if (!packet_) return;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    packet_ = nullptr;
    if (g_open == this) g_open = nullptr;
    p.cmd = g.cmd_;
    if (g_recorder_inline) {
        replay(p);
        const std::uint64_t n = g_published.load(std::memory_order_relaxed) + 1;
        g_published.store(n, std::memory_order_relaxed);
        g_done.store(n, std::memory_order_relaxed);
        g_replayed.store(n, std::memory_order_relaxed);
        return;
    }
    // A release store, not a locked add: that would wait here for this
    // draw's stores (its params blocks among them) to drain. Without the
    // fence the recorder may go to sleep with this packet unseen; the next
    // publish or the drain that needs it wakes it.
    g_published.store(g_published.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    if (g_sleeping.load(std::memory_order_relaxed)) g_published.notify_one();
}

void DrawCmds::update_sets(const VkWriteDescriptorSet* writes, std::size_t n) {
    if (!n) return;
    if (!packet_) {
        vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(n), writes, 0, nullptr);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    for (std::size_t i = 0; i < n; ++i) {
        VkWriteDescriptorSet w = writes[i];
        std::int32_t b = -1, im = -1;
        if (w.pBufferInfo) {
            b = static_cast<std::int32_t>(p.buffers.size());
            p.buffers.insert(p.buffers.end(), w.pBufferInfo, w.pBufferInfo + w.descriptorCount);
            w.pBufferInfo = nullptr;
        }
        if (w.pImageInfo) {
            im = static_cast<std::int32_t>(p.images.size());
            p.images.insert(p.images.end(), w.pImageInfo, w.pImageInfo + w.descriptorCount);
            w.pImageInfo = nullptr;
        }
        p.writes.push_back(w);
        p.write_buffer.push_back(b);
        p.write_image.push_back(im);
    }
}

void DrawCmds::take_sets(std::vector<VkWriteDescriptorSet>& writes, std::vector<VkDescriptorImageInfo>& images,
                         const std::vector<VkDescriptorBufferInfo>& buffers, const VkDescriptorBufferInfo* extra, std::size_t n_extra) {
    if (writes.empty()) return;
    if (!packet_) {
        vkUpdateDescriptorSets(g.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    const VkDescriptorBufferInfo* const b0 = buffers.data();
    const VkDescriptorBufferInfo* const b1 = b0 + buffers.size();
    const VkDescriptorImageInfo* const i0 = images.data();
    const VkDescriptorImageInfo* const i1 = i0 + images.size();
    // Taken only when every info is one of these and nothing came before
    // (the packet's writes are applied in the order they came).
    bool takeable = n_extra <= 2 && p.writes.empty() && p.taken_writes.empty();
    for (std::size_t k = 0; takeable && k < writes.size(); ++k) {
        const VkWriteDescriptorSet& w = writes[k];
        if (w.pBufferInfo) {
            takeable = (w.pBufferInfo >= b0 && w.pBufferInfo + w.descriptorCount <= b1) ||
                       (w.pBufferInfo >= extra && w.pBufferInfo + w.descriptorCount <= extra + n_extra);
        }
        if (takeable && w.pImageInfo) takeable = w.pImageInfo >= i0 && w.pImageInfo + w.descriptorCount <= i1;
        if (takeable && w.pTexelBufferView) takeable = false;
    }
    if (!takeable) {
        update_sets(writes.data(), writes.size());
        return;
    }
    p.taken_buffers.assign(buffers.begin(), buffers.end());
    for (std::size_t k = 0; k < n_extra; ++k) p.extra[k] = extra[k];
    for (VkWriteDescriptorSet& w : writes) {
        if (!w.pBufferInfo) continue;
        if (w.pBufferInfo >= b0 && w.pBufferInfo < b1) {
            w.pBufferInfo = p.taken_buffers.data() + (w.pBufferInfo - b0);
        } else {
            w.pBufferInfo = p.extra + (w.pBufferInfo - extra);
        }
    }
    p.taken_writes.swap(writes);  // image infos keep their buffer, and the writes' pointers into it
    p.taken_images.swap(images);
}

void DrawCmds::bind_pipeline(VkPipeline pipeline) {
    if (!packet_) {
        vkCmdBindPipeline(g_cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        return;
    }
    static_cast<DrawPacket*>(packet_)->pipeline = pipeline;
}

void DrawCmds::library_state(const DrawLibraryState& s, bool has_depth_bounds) {
    if (!packet_) {
        record_library_state(g_cmd(), s, has_depth_bounds);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.library = true;
    p.lib = s;
    p.has_depth_bounds = has_depth_bounds;
}

void DrawCmds::bind_sets(VkPipelineLayout layout, const VkDescriptorSet sets[2], const std::uint32_t* dynamic, bool bindless) {
    if (!packet_) {
        vkCmdBindDescriptorSets(g_cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, sets, dynamic ? 2 : 0, dynamic);
        if (bindless) vkCmdBindDescriptorSets(g_cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 3, 1, &g.bindless_set, 0, nullptr);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.bind_sets = true;
    p.bindless = bindless;
    p.layout = layout;
    p.sets[0] = sets[0];
    p.sets[1] = sets[1];
    p.dynamic_count = dynamic ? 2 : 0;
    if (dynamic) {
        p.dynamic[0] = dynamic[0];
        p.dynamic[1] = dynamic[1];
    }
}

void DrawCmds::push_buffers(VkPipelineLayout layout, const std::uint32_t* bindings, const VkDescriptorBufferInfo* infos, std::uint32_t n) {
    if (!n) return;
    if (!packet_) {
        VkWriteDescriptorSet writes[2 * gcn::kMaxBuffers];
        for (std::uint32_t k = 0; k < n; ++k) {
            writes[k] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[k].dstBinding = bindings[k];
            writes[k].descriptorCount = 1;
            writes[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[k].pBufferInfo = &infos[k];
        }
        g.cmd_push_descriptor_set(g_cmd(), VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 2, n, writes);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.layout = layout;
    p.pushes = n;
    std::memcpy(p.push_binding, bindings, n * sizeof(std::uint32_t));
    std::memcpy(p.push_infos, infos, n * sizeof(VkDescriptorBufferInfo));
}

void DrawCmds::viewport(const VkViewport& vp) {
    if (!packet_) {
        vkCmdSetViewport(g_cmd(), 0, 1, &vp);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.viewport = true;
    p.vp = vp;
}

void DrawCmds::scissor(const VkRect2D& sc) {
    if (!packet_) {
        vkCmdSetScissor(g_cmd(), 0, 1, &sc);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.scissor = true;
    p.sc = sc;
}

void DrawCmds::depth_bounds(float lo, float hi) {
    if (!packet_) {
        vkCmdSetDepthBounds(g_cmd(), lo, hi);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.bounds = true;
    p.depth_bounds[0] = lo;
    p.depth_bounds[1] = hi;
}

void DrawCmds::stencil(const std::uint32_t words[6]) {
    if (!packet_) {
        record_stencil_words(g_cmd(), words);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.stencil = true;
    std::memcpy(p.stencil_words, words, sizeof(p.stencil_words));
}

void DrawCmds::blend_constants(const float c[4]) {
    if (!packet_) {
        vkCmdSetBlendConstants(g_cmd(), c);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.blend = true;
    std::memcpy(p.blend_const, c, sizeof(p.blend_const));
}

void DrawCmds::index_buffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    if (!packet_) {
        vkCmdBindIndexBuffer(g_cmd(), buffer, offset, type);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.index = true;
    p.index_buffer = buffer;
    p.index_offset = offset;
    p.index_type = type;
}

void DrawCmds::vertex_buffers(std::uint32_t n, const VkBuffer* buffers, const VkDeviceSize* offsets) {
    if (!n) return;
    if (!packet_ || n > kMaxVertexBindings) {
        if (packet_) {
            publish();  // too many to pack: this draw records the rest in place
        }
        vkCmdBindVertexBuffers(g_cmd(), 0, n, buffers, offsets);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.vertex_buffers = n;
    std::memcpy(p.vb, buffers, n * sizeof(VkBuffer));
    std::memcpy(p.vb_offset, offsets, n * sizeof(VkDeviceSize));
}

void DrawCmds::draw(const DrawCall& call) {
    if (!packet_) {
        record_draw_call(g_cmd(), call);
        return;
    }
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    p.draw = true;
    p.call = call;
}

bool DrawCmds::end_rendering(bool barrier) {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.end_pass || p.nregions) return false;
    p.end_pass = true;
    p.end_pass_barrier = barrier;
    return true;
}

bool DrawCmds::pass_barrier() {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.pass_barrier || p.xfer_begin || p.xfer_end || p.ncopies || p.nregions) return false;
    p.pass_barrier = true;
    return true;
}

// The transfer batch's parts go between the pass end and the pass begin, so
// a packet takes them only while nothing after that point is in it.
bool DrawCmds::transfer_begin() {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.xfer_begin || p.xfer_end || p.ncopies || p.nregions) return false;
    p.xfer_begin = true;
    return true;
}

bool DrawCmds::copy_buffer(VkBuffer src, VkBuffer dst, const VkBufferCopy& region) {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.xfer_end || p.nregions || p.ncopies >= DrawPacket::kCopies) return false;
    p.copy_src[p.ncopies] = src;
    p.copy_dst[p.ncopies] = dst;
    p.copies[p.ncopies] = region;
    ++p.ncopies;
    return true;
}

bool DrawCmds::copy_order() {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.xfer_end || p.nregions || p.ncopies >= DrawPacket::kCopies) return false;
    p.copy_orders |= 1u << p.ncopies;
    return true;
}

bool DrawCmds::transfer_end() {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || p.xfer_end || p.nregions) return false;
    p.xfer_end = true;
    return true;
}

bool DrawCmds::copy_region(VkImage src, VkImage dst, std::uint32_t width, std::uint32_t height, bool dst_initialised) {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    // After a transfer batch's start its end must come first (replay order).
    if (!before_binds(p) || p.begin_pass || (p.xfer_begin && !p.xfer_end) || p.nregions >= DrawPacket::kRegionCopies) return false;
    p.regions[p.nregions++] = {src, dst, width, height, dst_initialised};
    return true;
}

bool DrawCmds::begin_rendering(const VkRenderingInfo& ri) {
    if (!packet_) return false;
    DrawPacket& p = *static_cast<DrawPacket*>(packet_);
    if (!before_binds(p) || p.begin_pass || ri.colorAttachmentCount > 8 || ri.pNext) return false;
    p.begin_pass = true;
    p.rendering = ri;
    for (std::uint32_t k = 0; k < ri.colorAttachmentCount; ++k) p.colors[k] = ri.pColorAttachments[k];
    p.has_depth = ri.pDepthAttachment != nullptr;
    if (p.has_depth) p.depth_att = *ri.pDepthAttachment;
    p.has_stencil = ri.pStencilAttachment != nullptr;
    if (p.has_stencil) p.stencil_att = *ri.pStencilAttachment;
    return true;
}

// ---- the commands themselves, shared by both paths ----

void record_library_state(VkCommandBuffer cmd, const DrawLibraryState& s, bool has_depth_bounds) {
    vkCmdSetCullMode(cmd, s.cull);
    vkCmdSetFrontFace(cmd, s.front);
    vkCmdSetDepthTestEnable(cmd, s.depth_test);
    vkCmdSetDepthWriteEnable(cmd, s.depth_write);
    vkCmdSetDepthCompareOp(cmd, s.depth_compare);
    if (has_depth_bounds) vkCmdSetDepthBoundsTestEnable(cmd, s.bounds_test);
    vkCmdSetStencilTestEnable(cmd, s.stencil_test);
    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, s.front_ops.failOp, s.front_ops.passOp, s.front_ops.depthFailOp, s.front_ops.compareOp);
    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, s.back_ops.failOp, s.back_ops.passOp, s.back_ops.depthFailOp, s.back_ops.compareOp);
    vkCmdSetDepthBias(cmd, s.bias_constant, s.bias_clamp, s.bias_slope);
    if (g.dynamic_depth_clamp) g.cmd_set_depth_clamp_enable(cmd, s.depth_clamp);
}

void record_stencil_words(VkCommandBuffer cmd, const std::uint32_t w[6]) {
    vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_BIT, w[0]);
    vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_BACK_BIT, w[1]);
    vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_BIT, w[2]);
    vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_BACK_BIT, w[3]);
    vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_BIT, w[4]);
    vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_BACK_BIT, w[5]);
}

// BBHOST_RENDER_MIN=1: a test client. Everything is recorded but the draws
// and dispatches themselves, so the game sees its labels, flips and
// readbacks and the GPU does next to nothing - several instances on one
// GPU for the online tests. The screen is whatever the clears leave.
const bool g_render_min = [] {
    const char* e = std::getenv("BBHOST_RENDER_MIN");
    return e && e[0] == '1';
}();

void record_draw_call(VkCommandBuffer cmd, const DrawCall& c) {
    if (g_render_min) return;
    if (c.query_pool) vkCmdBeginQuery(cmd, c.query_pool, c.query, glitch_query_flags());
    switch (c.kind) {
    case DrawCall::kIndexedIndirect: vkCmdDrawIndexedIndirect(cmd, c.buffer, c.offset, 1, 20); break;
    case DrawCall::kIndirect: vkCmdDrawIndirect(cmd, c.buffer, c.offset, 1, 16); break;
    case DrawCall::kIndexed: vkCmdDrawIndexed(cmd, c.count, c.instances, 0, c.vertex_offset, 0); break;
    case DrawCall::kDirect: vkCmdDraw(cmd, c.count, c.instances, static_cast<std::uint32_t>(c.vertex_offset), 0); break;
    }
    if (c.query_pool) vkCmdEndQuery(cmd, c.query_pool, c.query);
}

}  // namespace gpu
