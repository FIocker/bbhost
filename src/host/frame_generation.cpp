#include "host/frame_generation.h"

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>
#include "core/config.h"
#include "log.h"
#include "host/gpu.h"
#include "host/settings.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#include "../../tools/ngx_bridge/ngx_bridge.h"

namespace host {
namespace {

struct BridgeFg {
    bool loaded = false;
#if defined(_WIN32)
    HMODULE dll = nullptr;
#endif
    PFN_ngxb_fg_available fg_available = nullptr;
    PFN_ngxb_fg_create fg_create = nullptr;
    PFN_ngxb_fg_evaluate fg_evaluate = nullptr;
    PFN_ngxb_fg_max_generated fg_max_generated = nullptr;
    PFN_ngxb_fg_evaluate_index fg_evaluate_index = nullptr;
    PFN_ngxb_fg_release fg_release = nullptr;
} g_bridge_fg;

bool load_bridge_fg() {
    if (g_bridge_fg.loaded) return g_bridge_fg.fg_evaluate != nullptr;
    g_bridge_fg.loaded = true;
#if defined(_WIN32)
    HMODULE dll = GetModuleHandleW(L"ngx_bridge.dll");
    if (!dll) {
        const std::string path = config_exe_dir() + "/ngx_bridge.dll";
        std::wstring wpath(path.size() + 1, L'\0');
        wpath.resize(static_cast<std::size_t>(MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), static_cast<int>(wpath.size()))));
        if (!wpath.empty() && wpath.back() == L'\0') wpath.pop_back();
        dll = LoadLibraryExW(wpath.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    if (!dll) {
        host_log("framegen: ngx_bridge.dll not found; FG disabled");
        return false;
    }
    g_bridge_fg.dll = dll;
    auto get = [dll](const char* name) { return reinterpret_cast<void*>(GetProcAddress(dll, name)); };
    g_bridge_fg.fg_available = reinterpret_cast<PFN_ngxb_fg_available>(get("ngxb_fg_available"));
    g_bridge_fg.fg_create = reinterpret_cast<PFN_ngxb_fg_create>(get("ngxb_fg_create"));
    g_bridge_fg.fg_evaluate = reinterpret_cast<PFN_ngxb_fg_evaluate>(get("ngxb_fg_evaluate"));
    g_bridge_fg.fg_max_generated = reinterpret_cast<PFN_ngxb_fg_max_generated>(get("ngxb_fg_max_generated"));
    g_bridge_fg.fg_evaluate_index = reinterpret_cast<PFN_ngxb_fg_evaluate_index>(get("ngxb_fg_evaluate_index"));
    g_bridge_fg.fg_release = reinterpret_cast<PFN_ngxb_fg_release>(get("ngxb_fg_release"));
    if (!g_bridge_fg.fg_available || !g_bridge_fg.fg_create || !g_bridge_fg.fg_evaluate || !g_bridge_fg.fg_release) {
        host_log("framegen: ngx_bridge.dll missing FG entry points; FG disabled");
        g_bridge_fg.fg_evaluate = nullptr;
        return false;
    }
    return true;
#else
    return false;
#endif
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    return UINT32_MAX;
}

FgStats g_stats;
std::mutex g_stats_mu;

}  // namespace

bool fg_enabled() {
    return host_startup_settings().frame_generation;
}

bool fg_available() {
    if (!load_bridge_fg()) return false;
    return g_bridge_fg.fg_available ? g_bridge_fg.fg_available() != 0 : false;
}

FgStats fg_get_stats() {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    return g_stats;
}

void fg_note_real_presented() {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    ++g_stats.real_presented;
    if (g_stats.real_presented == 1 || g_stats.real_presented % 300 == 0)
        host_log("framegen: presented %llu real, %llu generated; %llu evaluations failed, %llu suppressed",
                 static_cast<unsigned long long>(g_stats.real_presented),
                 static_cast<unsigned long long>(g_stats.generated_presented),
                 static_cast<unsigned long long>(g_stats.evaluations_failed),
                 static_cast<unsigned long long>(g_stats.interpolation_disabled));
}

void fg_note_generated_presented() {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    ++g_stats.generated_presented;
}

void fg_note_interpolation_disabled() {
    std::lock_guard<std::mutex> lock(g_stats_mu);
    ++g_stats.interpolation_disabled;
    ++g_stats.frames_skipped;
}

FrameGenerator::~FrameGenerator() {
    shutdown();
}

bool FrameGenerator::init(VkInstance instance, VkPhysicalDevice phys, VkDevice device, std::uint32_t queue_family, VkQueue queue) {
    instance_ = instance;
    phys_ = phys;
    device_ = device;
    queue_family_ = queue_family;
    queue_ = queue;
    load_bridge_fg();

    if (!device_) return false;
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family_;
    if (vkCreateCommandPool(device_, &pci, nullptr, &eval_pool_) != VK_SUCCESS) return false;
    for (auto& frame : frames_) {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = eval_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &cai, &frame.cmd) != VK_SUCCESS) return false;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device_, &fci, nullptr, &frame.fence) != VK_SUCCESS) return false;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        if (vkCreateSemaphore(device_, &sci, nullptr, &frame.ready) != VK_SUCCESS) return false;
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(phys_, &props);
    std::uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, families.data());
    if (queue_family_ < n && families[queue_family_].timestampValidBits == 64) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = kFrameSlots * 2;
        if (vkCreateQueryPool(device_, &qi, nullptr, &times_) == VK_SUCCESS)
            timestamp_period_ns_ = props.limits.timestampPeriod;
    }
    return true;
}

void FrameGenerator::shutdown() {
    if (device_) {
        host_gpu_queue_lock();
        vkDeviceWaitIdle(device_);
        host_gpu_queue_unlock();
    }
    if (feature_created_ && g_bridge_fg.fg_release) g_bridge_fg.fg_release();
    feature_created_ = false;
    for (auto& frame : frames_) {
        for (auto& image : frame.generated) destroy_image(image);
        destroy_image(frame.real_copy); destroy_image(frame.input);
        if (frame.disable_mapped) vkUnmapMemory(device_, frame.disable_mem);
        if (frame.disable_buf) vkDestroyBuffer(device_, frame.disable_buf, nullptr);
        if (frame.disable_mem) vkFreeMemory(device_, frame.disable_mem, nullptr);
        if (frame.fence) vkDestroyFence(device_, frame.fence, nullptr);
        if (frame.ready) vkDestroySemaphore(device_, frame.ready, nullptr);
        frame = {};
    }
    if (times_) vkDestroyQueryPool(device_, times_, nullptr);
    times_ = VK_NULL_HANDLE;
    if (eval_pool_) vkDestroyCommandPool(device_, eval_pool_, nullptr);
    eval_pool_ = VK_NULL_HANDLE;
    width_ = height_ = 0;
    format_ = VK_FORMAT_UNDEFINED;
    device_ = VK_NULL_HANDLE;
    history_.clear();
}

bool FrameGenerator::create_image(OwnedImage& im, std::uint32_t w, std::uint32_t h, VkFormat fmt, VkImageUsageFlags usage) {
    destroy_image(im);
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device_, &ici, nullptr, &im.image) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, im.image, &req);
    uint32_t mtype = find_memory_type(phys_, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mtype == UINT32_MAX) return false;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mtype;
    if (vkAllocateMemory(device_, &mai, nullptr, &im.memory) != VK_SUCCESS) return false;
    if (vkBindImageMemory(device_, im.image, im.memory, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = im.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device_, &vci, nullptr, &im.view) != VK_SUCCESS) return false;

    im.format = fmt;
    im.width = w;
    im.height = h;
    return true;
}

void FrameGenerator::destroy_image(OwnedImage& im) {
    if (im.view) {
        vkDestroyImageView(device_, im.view, nullptr);
        im.view = VK_NULL_HANDLE;
    }
    if (im.image) {
        vkDestroyImage(device_, im.image, nullptr);
        im.image = VK_NULL_HANDLE;
    }
    if (im.memory) {
        vkFreeMemory(device_, im.memory, nullptr);
        im.memory = VK_NULL_HANDLE;
    }
    im.width = im.height = 0;
    im.format = VK_FORMAT_UNDEFINED;
}

bool FrameGenerator::ensure_feature(VkCommandBuffer cmd, std::uint32_t width, std::uint32_t height, VkFormat format) {
    if (ready() && width_ == width && height_ == height && format_ == format) return true;
    if (!load_bridge_fg() || !g_bridge_fg.fg_available || !g_bridge_fg.fg_available()) return false;

    // Even after a failed evaluation/released NGX feature, a previous real
    // frame's blit can still read our images. Retire that work before resizing.
    if (device_ && (frames_[0].generated[0].image || frames_[0].real_copy.image || frames_[0].input.image)) {
        host_gpu_queue_lock();
        vkDeviceWaitIdle(device_);
        host_gpu_queue_unlock();
    }
    if (feature_created_) {
        if (g_bridge_fg.fg_release) {
            g_bridge_fg.fg_release();
        }
        feature_created_ = false;
    }

    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const unsigned requested = std::clamp(host_startup_settings().frame_generation_factor, 2, 4) - 1;
    const unsigned supported = g_bridge_fg.fg_max_generated && g_bridge_fg.fg_evaluate_index ?
        g_bridge_fg.fg_max_generated() : 1;
    generated_count_ = requested <= supported ? requested : 1;
    if (generated_count_ != requested)
        host_log("framegen: requested %ux unsupported (maximum %ux); using 2x", requested + 1, supported + 1);
    for (auto& frame : frames_) {
        for (unsigned i = 0; i < frame.generated.size(); ++i) {
            if (i >= generated_count_) { destroy_image(frame.generated[i]); continue; }
            if (!create_image(frame.generated[i], width, height, format, usage)) {
                host_log("framegen: failed to create generated output image %u", i);
                return false;
            }
        }
        if (!create_image(frame.real_copy, width, height, format, usage)) {
            host_log("framegen: failed to create real copy image");
            return false;
        }
        if (!create_image(frame.input, width, height, format, usage)) return false;

        if (!frame.disable_buf) {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = 16;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            if (vkCreateBuffer(device_, &bci, nullptr, &frame.disable_buf) != VK_SUCCESS) return false;
            VkMemoryRequirements req{};
            vkGetBufferMemoryRequirements(device_, frame.disable_buf, &req);
            uint32_t mtype = find_memory_type(phys_, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (mtype == UINT32_MAX) {
                vkDestroyBuffer(device_, frame.disable_buf, nullptr); frame.disable_buf = VK_NULL_HANDLE;
                return false;
            }
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            mai.pNext = &flags;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = mtype;
            if (vkAllocateMemory(device_, &mai, nullptr, &frame.disable_mem) != VK_SUCCESS) {
                vkDestroyBuffer(device_, frame.disable_buf, nullptr); frame.disable_buf = VK_NULL_HANDLE;
                return false;
            }
            if (vkBindBufferMemory(device_, frame.disable_buf, frame.disable_mem, 0) != VK_SUCCESS ||
                vkMapMemory(device_, frame.disable_mem, 0, 16, 0, &frame.disable_mapped) != VK_SUCCESS) {
                vkDestroyBuffer(device_, frame.disable_buf, nullptr); frame.disable_buf = VK_NULL_HANDLE;
                vkFreeMemory(device_, frame.disable_mem, nullptr); frame.disable_mem = VK_NULL_HANDLE;
                return false;
            }
        }

        // Transition owned images to VK_IMAGE_LAYOUT_GENERAL
        VkImageMemoryBarrier barriers[5] = {};
        const unsigned barrier_count = generated_count_ + 2;
        for (unsigned i = 0; i < barrier_count; ++i) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barriers[i].srcQueueFamilyIndex = barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].image = i < generated_count_ ? frame.generated[i].image :
                i == generated_count_ ? frame.real_copy.image : frame.input.image;
            barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barriers[i].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        }
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, barrier_count, barriers);
    }

    uint32_t r = g_bridge_fg.fg_create(cmd, width, height, format);
    if (ngxb_failed(r)) {
        host_log("framegen: ngxb_fg_create failed (0x%08x) for %ux%u format %d", r, width, height, static_cast<int>(format));
        return false;
    }

    width_ = width;
    height_ = height;
    format_ = format;
    feature_created_ = true;
    recreate_ = false;
    history_.clear();
    host_log("framegen: DLSS Frame Generation %ux created for %ux%u format %d", generated_count_ + 1, width, height, static_cast<int>(format));
    return true;
}

bool FrameGenerator::evaluate(VkCommandBuffer cmd, VkImage color_image, VkImageView color_view, VkFormat color_format,
                              std::uint32_t width, std::uint32_t height, const gpu::DlssFgGuides& guides, unsigned slot) {
    auto& frame = frames_[slot];
    if (!feature_created_ || !g_bridge_fg.fg_evaluate) return false;
    if (!guides.valid || !guides.depth || !guides.motion) return false;

    // The game renders a logical picture in the top-left of its maximum-size
    // display allocation. NGX requires the actual picture extent and a matching
    // image, rather than the allocation's unused rows and columns.
    VkMemoryBarrier before_copy{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before_copy.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    before_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &before_copy, 0, nullptr, 0, nullptr);
    VkImageCopy crop{};
    crop.srcSubresource = crop.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    crop.extent = {width, height, 1};
    vkCmdCopyImage(cmd, color_image, VK_IMAGE_LAYOUT_GENERAL, frame.input.image,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &crop);
    (void)color_view;
    (void)color_format;

    if (times_) {
        vkCmdResetQueryPool(cmd, times_, slot * 2, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, times_, slot * 2);
    }
    const bool reset = guides.reset || history_.needs_reset(guides.render_frame_index);
    if (frame.disable_mapped) {
        *static_cast<uint32_t*>(frame.disable_mapped) = reset ? 1u : 0u;
    }

    NgxbGenerate gen{};
    gen.color = {frame.input.image, frame.input.view, frame.input.format, width, height, VK_IMAGE_ASPECT_COLOR_BIT};
    gen.real = {frame.real_copy.image, frame.real_copy.view, frame.real_copy.format, width, height, VK_IMAGE_ASPECT_COLOR_BIT};
    gen.depth = {guides.depth, guides.depth_view, guides.depth_format, guides.width, guides.height, VK_IMAGE_ASPECT_COLOR_BIT};
    gen.motion = {guides.motion, guides.motion_view, guides.motion_format, guides.width, guides.height, VK_IMAGE_ASPECT_COLOR_BIT};
    if (guides.hudless) {
        const std::uint32_t hw = guides.hudless_width ? guides.hudless_width : guides.width;
        const std::uint32_t hh = guides.hudless_height ? guides.hudless_height : guides.height;
        gen.hudless = {guides.hudless, guides.hudless_view, guides.hudless_format, hw, hh, VK_IMAGE_ASPECT_COLOR_BIT};
    }
    gen.camera = guides.camera;
    gen.disable_interpolation = frame.disable_buf;
    gen.reset = reset;

    VkMemoryBarrier pre_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    pre_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    pre_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    uint32_t r = 1;
    for (unsigned i = 0; i < generated_count_; ++i) {
        const auto& image = frame.generated[i];
        gen.output = {image.image, image.view, image.format, width, height, VK_IMAGE_ASPECT_COLOR_BIT};
        r = g_bridge_fg.fg_evaluate_index ? g_bridge_fg.fg_evaluate_index(cmd, &gen,
            generated_count_, i + 1, guides.render_frame_index) : g_bridge_fg.fg_evaluate(cmd, &gen);
        if (ngxb_failed(r)) break;
        // Every subframe shares the source inputs and NGX's feature state. Make
        // writes visible before requesting the next temporal position.
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);
    }
    {
        std::lock_guard<std::mutex> lock(g_stats_mu);
        ++g_stats.evaluations;
        if (ngxb_failed(r)) ++g_stats.evaluations_failed;
    }
    if (ngxb_failed(r)) {
        static int fail_count = 0;
        if (++fail_count <= 5) host_log("framegen: evaluate failed (0x%08x)", r);
        return false;
    }

    VkMemoryBarrier post_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    post_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    post_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
                                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    if (times_) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, times_, slot * 2 + 1);
    return true;
}

bool FrameGenerator::is_interpolation_disabled(unsigned slot) {
    const auto& frame = frames_[slot];
    if (!frame.disable_mapped) return false;
    // Memory is HOST_COHERENT and the evaluation fence has completed.
    return *static_cast<const uint32_t*>(frame.disable_mapped) != 0;
}

bool FrameGenerator::evaluate_submit(VkImage color_image, VkImageView color_view, VkFormat color_format,
                                     std::uint32_t width, std::uint32_t height, const gpu::DlssFgGuides& guides, unsigned slot) {
    auto& frame = frames_[slot];
    if (!frame.cmd || !frame.fence) return false;
    if (frame.pending) {
        if (vkGetFenceStatus(device_, frame.fence) != VK_SUCCESS) return false;
        frame.pending = false;
    }

    vkResetCommandBuffer(frame.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(frame.cmd, &bi) != VK_SUCCESS) return false;

    if (!ensure_feature(frame.cmd, width, height, color_format)) {
        vkEndCommandBuffer(frame.cmd);
        return false;
    }

    if (!evaluate(frame.cmd, color_image, color_view, color_format, width, height, guides, slot)) {
        vkEndCommandBuffer(frame.cmd);
        // This recording is discarded; recreate on retry so an image whose
        // transition never ran cannot be mistaken for one in GENERAL layout.
        // Previously submitted frames may still use NGX's feature state. The
        // producer drains their leases before ensure_feature releases it.
        recreate_ = true;
        return false;
    }
    if (vkEndCommandBuffer(frame.cmd) != VK_SUCCESS) return false;

    vkResetFences(device_, 1, &frame.fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &frame.cmd;
    // A frame set discarded before presentation still has a signalled binary
    // semaphore. Consume that signal before re-signalling it on slot reuse.
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkSemaphore wait = frame.ready_unconsumed ? frame.ready : VK_NULL_HANDLE;
    si.waitSemaphoreCount = wait ? 1 : 0;
    si.pWaitSemaphores = wait ? &wait : nullptr;
    si.pWaitDstStageMask = wait ? &wait_stage : nullptr;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &frame.ready;
    frame.ticket = host_gpu_submit_presenter(frame.cmd, wait, wait_stage, frame.ready, frame.fence);
    VkResult res = VK_SUCCESS;
    if (!frame.ticket) {
        host_gpu_queue_lock();
        res = vkQueueSubmit(queue_, 1, &si, frame.fence);
        host_gpu_queue_unlock();
    }
    frame.pending = res == VK_SUCCESS;
    if (frame.pending) { history_.submitted(guides.render_frame_index); frame.ready_unconsumed = true; }
    else history_.clear();
    return frame.pending;
}

bool FrameGenerator::wait_evaluation(std::uint64_t timeout_ns, unsigned slot) {
    auto& frame = frames_[slot];
    if (!frame.fence || !frame.pending) return false;
    host_gpu_wait_submitted(frame.ticket);
    const bool done = vkWaitForFences(device_, 1, &frame.fence, VK_TRUE, timeout_ns) == VK_SUCCESS;
    // The producer clears pending after observing this fence before slot reuse.
    return done;
}

bool FrameGenerator::evaluation_finished(unsigned slot) const {
    return vkGetFenceStatus(device_, frames_[slot].fence) == VK_SUCCESS;
}

bool FrameGenerator::slot_available(unsigned slot) const {
    return !frames_[slot].pending || evaluation_finished(slot);
}

bool FrameGenerator::evaluation_times(unsigned slot, std::uint64_t* source_ns, std::uint64_t* ready_ns) const {
    if (!times_) return false;
    std::uint64_t values[2]{};
    if (vkGetQueryPoolResults(device_, times_, slot * 2, 2, sizeof(values), values,
            sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return false;
    *source_ns = static_cast<std::uint64_t>(values[0] * timestamp_period_ns_);
    *ready_ns = static_cast<std::uint64_t>(values[1] * timestamp_period_ns_);
    return *ready_ns >= *source_ns;
}

FrameGenerator& fg_get() {
    static FrameGenerator instance;
    return instance;
}

}  // namespace host
