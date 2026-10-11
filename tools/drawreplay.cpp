// drawreplay: run a BBHOST_CAPTURE_DRAW capture on its own Vulkan device from
// its manifest and blobs alone.
//
//   drawreplay <capture-dir> [--out DIR] [--atol X] [--rtol Y] [--ps FILE.spv] [--vs FILE.spv]
//              [--no-live] [--validate]
//
// Runs, each into fresh targets initialised from the captured contents:
//   A, B       the captured modules twice; A must equal B.
//   live       A against the game's own output of the draw. This is the check
//              that the capture holds everything the draw's output depends on.
//   sentinel   the page table's sink page filled with varied values: a
//              difference from A means the shaders read guest memory the
//              capture lacks. Supplementary only: a missing vertex page whose
//              sentinel geometry is clipped or depth-rejected leaves the
//              output unchanged, and only `live` catches it.
//   perturbed  one texel of the first colour target's initial contents changed:
//              the comparison must flag exactly that texel. A perturbed copy of
//              A's output checks the comparator the same way.
//   variant    --ps: the same draw with another pixel shader, against A.
// --specialize: every run's stages take the specialization constants bbhost's
// optimized relink gives them (render.cpp relink_pipeline): the vertex
// formats as constants for a vertex shader that reads them from its params,
// and the indexed loads' words 3 and strides from the stage params.
// A draw that took Vulkan vertex input replays with the captured bindings'
// bytes as vertex buffers and the captured bindings and attributes.
// Every run restores the guest pages and buffers from the blobs and checks
// afterwards that the draw left them unchanged - except a storage buffer the
// shaders store to (the manifest's "written": translate.cpp store_bound): its
// contents after each run are an output, compared byte for byte between A and
// B and between A and the variant like the targets. Writes report.json, the runs'
// raw target contents and error heatmaps to --out (default <capture>/replay-N).
// Exit 0: every check passed; 1: a check failed; 2: bad capture or Vulkan error.
#include "core/sha256.h"
#include "gcn/translate.h"
#include "replay/compare.h"
#include "replay/json.h"

#include <vulkan/vulkan.h>
#if defined(DRAWREPLAY_HAVE_SPIRV_TOOLS)
#include <spirv-tools/libspirv.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t kPage = 1ull << gcn::kPageShift;
// The sentinel run's sink page: varied finite floats in [-1, 1]. A single
// repeated word is not enough - vertices read from an uncaptured page would
// all be equal, the draw would cover nothing, and the run would match A.
std::uint32_t sentinel_word(std::uint64_t k) {
    const float f = static_cast<float>(static_cast<int>(((k * 2654435761u) >> 8) % 2001) - 1000) / 1000.0f;
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    return bits;
}

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error(why); }

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) fail(std::string(what) + " failed (VkResult " + std::to_string(r) + ")");
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot read " + path.string());
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    f.seekg(0);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (size > 0 && !f.read(reinterpret_cast<char*>(data.data()), size)) fail("cannot read " + path.string());
    return data;
}

void write_file(const fs::path& path, const std::uint8_t* data, std::size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f || !f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes))) fail("cannot write " + path.string());
}

std::int64_t integer(const json::Value& v, const char* what) {
    if (v.type != json::Value::Type::Number || v.number != std::floor(v.number) || std::fabs(v.number) > 9.0e15) {
        fail(std::string("manifest value '") + what + "' must be an integer");
    }
    return static_cast<std::int64_t>(v.number);
}

std::uint32_t u32_of(const json::Value& v, const char* what) {
    const std::uint64_t x = json::as_u64(v, what);
    if (x > 0xffffffffull) fail(std::string("manifest value '") + what + "' exceeds 32 bits");
    return static_cast<std::uint32_t>(x);
}

bool is_null(const json::Value& obj, const char* key) {
    const json::Value* v = obj.find(key);
    return !v || v->type == json::Value::Type::Null;
}

VkImageAspectFlags full_aspect(VkFormat f) {
    switch (f) {
    case VK_FORMAT_D16_UNORM: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_X8_D24_UNORM_PACK32: return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_S8_UINT: return VK_IMAGE_ASPECT_STENCIL_BIT;
    case VK_FORMAT_D16_UNORM_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default: return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// ---- capture --------------------------------------------------------------------

struct Capture {
    fs::path dir;
    json::Value m;
    std::map<std::string, std::vector<std::uint8_t>> blobs;

    const std::vector<std::uint8_t>& blob(const std::string& name) const {
        auto it = blobs.find(name);
        if (it == blobs.end()) fail("manifest refers to blob '" + name + "', which it does not list");
        return it->second;
    }
};

Capture load_capture(const fs::path& dir) {
    Capture c;
    c.dir = dir;
    const std::vector<std::uint8_t> text = read_file(dir / "manifest.json");
    std::string error;
    if (!json::parse(std::string(text.begin(), text.end()), c.m, error)) fail("manifest.json: " + error);
    if (json::str(c.m, "schema") != "bbhost-draw-capture") fail("manifest.json is not a bbhost draw capture");
    if (json::u64(c.m, "version") != 1) fail("unsupported capture version " + std::to_string(json::u64(c.m, "version")));
    for (const auto& [name, b] : json::member(c.m, "blobs").object) {
        const bool plain = !name.empty() && std::all_of(name.begin(), name.end(), [](char ch) {
            return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_';
        });
        const std::string& file = json::str(b, "file");
        if (!plain || file != "blobs/" + name + ".bin") fail("blob '" + name + "' has an unexpected name or file " + file);
        std::vector<std::uint8_t> data = read_file(dir / file);
        if (data.size() != json::u64(b, "bytes")) {
            fail("blob " + file + " holds " + std::to_string(data.size()) + " bytes; the manifest says " + std::to_string(json::u64(b, "bytes")));
        }
        if (sha256_hex(data.data(), data.size()) != json::str(b, "sha256")) fail("blob " + file + " does not match its SHA-256");
        c.blobs.emplace(name, std::move(data));
    }
    return c;
}

std::vector<std::uint32_t> words_of(const std::vector<std::uint8_t>& bytes, const std::string& what) {
    if (bytes.size() % 4 || bytes.size() < 20) fail(what + " is not a SPIR-V module");
    std::vector<std::uint32_t> w(bytes.size() / 4);
    std::memcpy(w.data(), bytes.data(), bytes.size());
    if (w[0] != 0x07230203u) fail(what + " does not start with the SPIR-V magic number");
    return w;
}

// Whether a module decorates anything into descriptor set `set` (bbhost puts
// constant buffers in a push-descriptor set 2 of their own, vertex stage from
// binding 112 and pixel stage from 128; older captures kept them in the
// stage's own set).
bool spirv_uses_set(const std::vector<std::uint32_t>& w, std::uint32_t set) {
    for (std::size_t i = 5; i < w.size();) {
        const std::uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (n == 0) break;
        if (op == 71 /* OpDecorate */ && n >= 4 && w[i + 2] == 34 /* DescriptorSet */ && w[i + 3] == set) return true;
        i += n;
    }
    return false;
}

std::string validate_spirv(const std::vector<std::uint32_t>& words) {
#if defined(DRAWREPLAY_HAVE_SPIRV_TOOLS)
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& pos, const char* message) {
        if (msg.empty()) msg = "word " + std::to_string(pos.index) + ": " + message;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);  // as gcn2spv: VK_KHR_maintenance8 per-lane offsets
    if (!tools.Validate(words.data(), words.size(), vo)) return msg.empty() ? "invalid" : msg;
    return "valid";
#else
    (void)words;
    return "not checked (built without SPIRV-Tools)";
#endif
}

// ---- device ----------------------------------------------------------------------

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* map = nullptr;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageCreateInfo info{};
};

VkBool32 VKAPI_PTR on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
                                 const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
    auto* messages = static_cast<std::vector<std::string>*>(user);
    if (messages->size() < 64 && data && data->pMessage) messages->push_back(data->pMessage);
    return VK_FALSE;
}

class Device {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem{};
    bool depth_bounds = false;
    bool gpl = false;                  // VK_EXT_graphics_pipeline_library (--library)
    float timestamp_period_ns = 0.0f;  // 0: the queue has no timestamp queries
    std::string name;
    std::vector<std::string> validation;

    void init(const std::string& prefer, bool validate) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "drawreplay";
        app.apiVersion = VK_API_VERSION_1_3;
        std::vector<const char*> layers, exts;
        if (validate) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        ici.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
        ici.ppEnabledLayerNames = layers.data();
        ici.enabledExtensionCount = static_cast<std::uint32_t>(exts.size());
        ici.ppEnabledExtensionNames = exts.data();
        check(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
        if (validate) {
            VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
            mci.pfnUserCallback = on_validation;
            mci.pUserData = &validation;
            // The loader resolves extension entry points per instance.
            const auto create = reinterpret_cast<VkResult (*)(VkInstance, const VkDebugUtilsMessengerCreateInfoEXT*, const VkAllocationCallbacks*,
                                                               VkDebugUtilsMessengerEXT*)>(vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
            if (!create) fail("the validation layer's debug messenger is unavailable");
            check(create(instance, &mci, nullptr, &messenger), "vkCreateDebugUtilsMessengerEXT");
        }
        std::uint32_t n = 0;
        check(vkEnumeratePhysicalDevices(instance, &n, nullptr), "vkEnumeratePhysicalDevices");
        std::vector<VkPhysicalDevice> devices(n);
        check(vkEnumeratePhysicalDevices(instance, &n, devices.data()), "vkEnumeratePhysicalDevices");
        std::uint32_t family = 0;
        for (int pass = 0; pass < 2 && !phys; ++pass) {
            for (VkPhysicalDevice d : devices) {
                VkPhysicalDeviceProperties p{};
                vkGetPhysicalDeviceProperties(d, &p);
                if (p.apiVersion < VK_API_VERSION_1_3 || (pass == 0 && prefer != p.deviceName)) continue;
                std::uint32_t nq = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
                std::vector<VkQueueFamilyProperties> qs(nq);
                vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, qs.data());
                for (std::uint32_t i = 0; i < nq; ++i) {
                    if (qs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                        phys = d;
                        family = i;
                        name = p.deviceName;
                        if (qs[i].timestampValidBits) timestamp_period_ns = p.limits.timestampPeriod;
                        break;
                    }
                }
                if (phys) break;
            }
        }
        if (!phys) fail("no Vulkan 1.3 device with a graphics queue");
        vkGetPhysicalDeviceMemoryProperties(phys, &mem);
        VkPhysicalDeviceFeatures supported{};
        vkGetPhysicalDeviceFeatures(phys, &supported);
        depth_bounds = supported.depthBounds == VK_TRUE;
        std::uint32_t next = 0;
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, nullptr);
        std::vector<VkExtensionProperties> props(next);
        vkEnumerateDeviceExtensionProperties(phys, nullptr, &next, props.data());
        std::vector<const char*> dext;
        bool gpl_ext = false, lib_ext = false;
        for (const VkExtensionProperties& e : props) {
            if (std::strcmp(e.extensionName, "VK_KHR_maintenance8") == 0) dext.push_back("VK_KHR_maintenance8");
            if (std::strcmp(e.extensionName, "VK_EXT_graphics_pipeline_library") == 0) gpl_ext = true;
            if (std::strcmp(e.extensionName, "VK_KHR_pipeline_library") == 0) lib_ext = true;
        }
        gpl = gpl_ext && lib_ext;
        if (gpl) {  // bbhost's BBHOST_PIPELINE_LIBRARY path, replayed with --library
            dext.push_back("VK_KHR_pipeline_library");
            dext.push_back("VK_EXT_graphics_pipeline_library");
        }
        // The feature set bbhost creates its device with (src/host/gpu.cpp).
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        f13.dynamicRendering = VK_TRUE;
        f13.synchronization2 = VK_TRUE;
        VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT fgpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
        fgpl.graphicsPipelineLibrary = VK_TRUE;
        if (gpl) f13.pNext = &fgpl;
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.pNext = &f13;
        f12.bufferDeviceAddress = VK_TRUE;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f12;
        f2.features.shaderInt64 = VK_TRUE;
        f2.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
        f2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
        f2.features.geometryShader = VK_TRUE;
        f2.features.independentBlend = VK_TRUE;
        f2.features.fillModeNonSolid = VK_TRUE;
        f2.features.depthClamp = VK_TRUE;
        f2.features.dualSrcBlend = VK_TRUE;
        f2.features.shaderClipDistance = VK_TRUE;
        f2.features.shaderCullDistance = VK_TRUE;
        f2.features.samplerAnisotropy = VK_TRUE;
        f2.features.fragmentStoresAndAtomics = VK_TRUE;
        f2.features.vertexPipelineStoresAndAtomics = VK_TRUE;
        f2.features.depthBounds = supported.depthBounds;
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<std::uint32_t>(dext.size());
        dci.ppEnabledExtensionNames = dext.data();
        check(vkCreateDevice(phys, &dci, nullptr, &device), "vkCreateDevice (bbhost's feature set)");
        vkGetDeviceQueue(device, family, 0, &queue);
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family;
        check(vkCreateCommandPool(device, &pci, nullptr, &pool), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &cai, &cmd), "vkAllocateCommandBuffers");
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(device, &fci, nullptr, &fence), "vkCreateFence");
    }

    ~Device() {
        if (device) {
            vkDeviceWaitIdle(device);
            vkDestroyFence(device, fence, nullptr);
            vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (messenger) {
            const auto destroy = reinterpret_cast<void (*)(VkInstance, VkDebugUtilsMessengerEXT, const VkAllocationCallbacks*)>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy) destroy(instance, messenger, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) const {
        for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        fail("no memory type with properties " + std::to_string(flags));
    }

    // Host-visible, device-addressable.
    Buffer buffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        Buffer b;
        b.size = std::max<VkDeviceSize>(size, 4);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = b.size;
        bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device, &bci, nullptr, &b.buffer), "vkCreateBuffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &flags;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkAllocateMemory(device, &mai, nullptr, &b.memory), "vkAllocateMemory (buffer)");
        check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");
        check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
        VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        bai.buffer = b.buffer;
        b.address = vkGetBufferDeviceAddress(device, &bai);
        std::memset(b.map, 0, b.size);
        return b;
    }

    void destroy(Buffer& b) {
        if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(device, b.memory, nullptr);
        b = Buffer{};
    }

    Image image(const VkImageCreateInfo& info) {
        Image img;
        img.info = info;
        check(vkCreateImage(device, &info, nullptr, &img.image), "vkCreateImage");
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, img.image, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device, &mai, nullptr, &img.memory), "vkAllocateMemory (image)");
        check(vkBindImageMemory(device, img.image, img.memory, 0), "vkBindImageMemory");
        return img;
    }

    void destroy(Image& img) {
        if (img.image) vkDestroyImage(device, img.image, nullptr);
        if (img.memory) vkFreeMemory(device, img.memory, nullptr);
        img = Image{};
    }

    void submit(const std::function<void(VkCommandBuffer)>& record) {
        check(vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        record(cmd);
        check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        check(vkResetFences(device, 1, &fence), "vkResetFences");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        check(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit");
        check(vkWaitForFences(device, 1, &fence, VK_TRUE, 60'000'000'000ull), "waiting for the replay submission");
    }
};

void transition(VkCommandBuffer cmd, const Image& img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img.image;
    b.subresourceRange = {full_aspect(img.info.format), 0, img.info.mipLevels, 0, img.info.arrayLayers};
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

struct Subresource {
    std::uint32_t level = 0, layer = 0, w = 1, h = 1, d = 1;
    std::uint64_t offset = 0, bytes = 0;
};

// Copies `data` into the image (which must be in `from` layout) and leaves it
// in GENERAL.
void upload(Device& dev, const Image& img, VkImageAspectFlags aspect, const std::vector<std::uint8_t>& data,
            const std::vector<Subresource>& subs, VkImageLayout from) {
    Buffer staging = dev.buffer(data.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(staging.map, data.data(), data.size());
    std::vector<VkBufferImageCopy> regions;
    for (const Subresource& s : subs) {
        if (s.offset > data.size() || s.bytes > data.size() - s.offset) fail("subresource lies outside its blob");
        if (s.bytes != replay::subresource_bytes(img.info.format, aspect, s.w, s.h, s.d)) fail("subresource size does not match its format and extent");
        if (s.level >= img.info.mipLevels || s.layer >= img.info.arrayLayers) fail("subresource outside the image");
        VkBufferImageCopy r{};
        r.bufferOffset = s.offset;
        r.imageSubresource = {aspect, s.level, s.layer, 1};
        r.imageExtent = {s.w, s.h, s.d};
        regions.push_back(r);
    }
    dev.submit([&](VkCommandBuffer cmd) {
        transition(cmd, img, from, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(cmd, staging.buffer, img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<std::uint32_t>(regions.size()),
                               regions.data());
        transition(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    });
    dev.destroy(staging);
}

VkImageCreateInfo image_info_from(const json::Value& j) {
    VkImageCreateInfo i{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    i.flags = json::u32(j, "flags") & (VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
    i.imageType = static_cast<VkImageType>(json::u32(j, "type"));
    i.format = static_cast<VkFormat>(json::u32(j, "format"));
    const auto& ext = json::arr(j, "extent");
    if (ext.size() != 3) fail("image extent must have three members");
    i.extent = {u32_of(ext[0], "extent"), u32_of(ext[1], "extent"), u32_of(ext[2], "extent")};
    i.mipLevels = json::u32(j, "mip_levels");
    i.arrayLayers = json::u32(j, "array_layers");
    if (json::u32(j, "samples") != 1) fail("multisampled images are not supported");
    if (!i.extent.width || !i.extent.height || !i.extent.depth || i.extent.width > 16384 || i.extent.height > 16384 ||
        i.extent.depth > 2048 || !i.mipLevels || i.mipLevels > 16 || !i.arrayLayers || i.arrayLayers > 2048) {
        fail("image dimensions out of range");
    }
    i.samples = VK_SAMPLE_COUNT_1_BIT;
    i.tiling = VK_IMAGE_TILING_OPTIMAL;
    i.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    i.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return i;
}

VkImageViewCreateInfo view_info_from(const json::Value& j, VkImage image) {
    VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    v.image = image;
    v.viewType = static_cast<VkImageViewType>(json::u32(j, "type"));
    v.format = static_cast<VkFormat>(json::u32(j, "format"));
    const auto& comp = json::arr(j, "components");
    if (comp.size() != 4) fail("view components must have four members");
    v.components = {static_cast<VkComponentSwizzle>(u32_of(comp[0], "components")), static_cast<VkComponentSwizzle>(u32_of(comp[1], "components")),
                    static_cast<VkComponentSwizzle>(u32_of(comp[2], "components")), static_cast<VkComponentSwizzle>(u32_of(comp[3], "components"))};
    v.subresourceRange = {json::u32(j, "aspect"), json::u32(j, "base_level"), json::u32(j, "levels"), json::u32(j, "base_layer"),
                          json::u32(j, "layers")};
    return v;
}

std::vector<Subresource> subresources_from(const json::Value& list) {
    std::vector<Subresource> out;
    for (const json::Value& s : list.array) {
        Subresource r;
        r.level = json::u32(s, "level");
        r.layer = json::u32(s, "layer");
        const auto& ext = json::arr(s, "extent");
        if (ext.size() != 3) fail("subresource extent must have three members");
        r.w = u32_of(ext[0], "extent");
        r.h = u32_of(ext[1], "extent");
        r.d = u32_of(ext[2], "extent");
        r.offset = json::u64(s, "offset");
        r.bytes = json::u64(s, "bytes");
        out.push_back(r);
    }
    return out;
}

// ---- replay ------------------------------------------------------------------------

struct Output {
    std::string target;  // color<slot>, depth, stencil
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t aspect = 0;
    std::uint32_t width = 0, height = 0;
    std::vector<std::uint8_t> bytes;
};

struct RunResult {
    std::string name;
    std::vector<Output> outputs;
    std::vector<std::string> writes;  // guest memory or buffers the draw changed
    // The written storage buffers' contents after the run, in stage and binding
    // order (StageData::buffer_written).
    std::vector<std::vector<std::uint8_t>> stored;
    double gpu_ms = -1.0;             // RunOptions::timed: GPU time of the draws
};

struct RunOptions {
    std::string name;
    VkPipeline pipeline = VK_NULL_HANDLE;
    bool sentinel = false;  // fill the sink page with sentinel_word()
    bool perturb_initial = false;
    std::uint32_t px = 0, py = 0;
    std::uint32_t repeat = 1;  // --bench: the draw recorded this many times
    bool timed = false;        // timestamps around the draws
    bool library = false;      // the pipeline is linked from libraries: independent sets, depth/stencil/cull state set per draw
};

struct ColorTarget {
    int slot = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0, height = 0;
    std::string initial, live;
};

class Replayer {
public:
    Replayer(const Capture& cap, Device& dev) : cap_(cap), dev_(dev) {
        const json::Value& m = cap.m;
        load_targets();
        build_memory();
        build_layout();
        build_resources();
        build_stages();
        const json::Value& draw = json::member(m, "draw");
        if (json::flag(draw, "indexed")) {
            const auto& indices = cap.blob(json::str(draw, "index_blob"));
            index_ = dev.buffer(indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            std::memcpy(index_.map, indices.data(), indices.size());
        }
        // A draw that took vertex input: its bindings' bytes as vertex buffers.
        const json::Value* vinput = m.find("vertex_input");
        if (vinput && vinput->type == json::Value::Type::Object) {
            for (const json::Value& b : json::arr(*vinput, "bindings")) {
                if (json::u32(b, "binding") != vertex_buffers_.size()) fail("vertex input bindings are not numbered in order");
                const auto& bytes = cap.blob(json::str(b, "blob"));
                Buffer vb = dev.buffer(bytes.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
                std::memcpy(vb.map, bytes.data(), bytes.size());
                vertex_buffers_.push_back(vb);
                vertex_bindings_.push_back({json::u32(b, "binding"), json::u32(b, "stride"), VK_VERTEX_INPUT_RATE_VERTEX});
            }
            for (const json::Value& a : json::arr(*vinput, "attributes")) {
                vertex_attributes_.push_back(
                    {json::u32(a, "location"), json::u32(a, "binding"), static_cast<VkFormat>(json::u32(a, "format")), json::u32(a, "offset")});
            }
        }
    }

    ~Replayer() {
        vkDeviceWaitIdle(dev_.device);
        for (VkPipeline p : pipelines_) vkDestroyPipeline(dev_.device, p, nullptr);
        for (VkShaderModule s : modules_) vkDestroyShaderModule(dev_.device, s, nullptr);
        for (auto& [id, v] : views_) vkDestroyImageView(dev_.device, v, nullptr);
        for (auto& [id, img] : images_) dev_.destroy(img);
        for (auto& [id, s] : samplers_) vkDestroySampler(dev_.device, s, nullptr);
        if (pool_) vkDestroyDescriptorPool(dev_.device, pool_, nullptr);
        if (layout_) vkDestroyPipelineLayout(dev_.device, layout_, nullptr);
        if (set_layout_) vkDestroyDescriptorSetLayout(dev_.device, set_layout_, nullptr);
        if (cb_set_layout_) vkDestroyDescriptorSetLayout(dev_.device, cb_set_layout_, nullptr);
        for (auto& st : stages_) {
            dev_.destroy(st.params);
            for (Buffer& b : st.buffers) dev_.destroy(b);
        }
        dev_.destroy(index_);
        for (Buffer& b : vertex_buffers_) dev_.destroy(b);
        dev_.destroy(pages_);
        dev_.destroy(sink_);
        dev_.destroy(sink_l2_);
        dev_.destroy(l1_);
        for (auto& [i, b] : l2_) dev_.destroy(b);
    }

    const std::vector<ColorTarget>& colors() const { return colors_; }
    std::map<std::string, std::string> spirv_checks;

    // The captured modules, or the captured draw with `ps` as its pixel shader.
    VkPipeline pipeline(const std::vector<std::uint32_t>* ps_override, const std::vector<std::uint32_t>* vs_override = nullptr, bool library = false) {
        const json::Value& m = cap_.m;
        const json::Value& p = json::member(m, "pipeline");
        const auto& stages = json::arr(m, "stages");
        if (stages.size() != 2) fail("a capture has a vertex and a pixel stage entry");
        std::vector<VkPipelineShaderStageCreateInfo> infos;
        const auto add = [&](VkShaderStageFlagBits bit, const std::vector<std::uint32_t>& words, const std::string& label) {
            spirv_checks[label] = validate_spirv(words);
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smci.codeSize = words.size() * 4;
            smci.pCode = words.data();
            VkShaderModule module = VK_NULL_HANDLE;
            check(vkCreateShaderModule(dev_.device, &smci, nullptr, &module), ("vkCreateShaderModule " + label).c_str());
            modules_.push_back(module);
            VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            si.stage = bit;
            si.module = module;
            si.pName = "main";
            infos.push_back(si);
        };
        if (!json::flag(stages[0], "present")) fail("the capture has no vertex shader");
        if (specialize) specialization();
        if (vs_override) {
            add(VK_SHADER_STAGE_VERTEX_BIT, *vs_override, "vertex (variant)");
        } else {
            add(VK_SHADER_STAGE_VERTEX_BIT, words_of(cap_.blob(json::str(stages[0], "spirv")), "vertex SPIR-V"), "vertex");
        }
        const json::Value& shaders = json::member(m, "shaders");
        if (!is_null(shaders, "geometry")) {
            add(VK_SHADER_STAGE_GEOMETRY_BIT, words_of(cap_.blob(json::str(shaders, "geometry")), "geometry SPIR-V"), "geometry");
        }
        if (ps_override) {
            add(VK_SHADER_STAGE_FRAGMENT_BIT, *ps_override, "pixel (variant)");
        } else if (json::flag(stages[1], "present")) {
            add(VK_SHADER_STAGE_FRAGMENT_BIT, words_of(cap_.blob(json::str(stages[1], "spirv")), "pixel SPIR-V"), "pixel");
        }
        if (specialize) {
            for (VkPipelineShaderStageCreateInfo& si : infos) {
                if (si.stage == VK_SHADER_STAGE_VERTEX_BIT) si.pSpecializationInfo = &vs_spec_info_;
                if (si.stage == VK_SHADER_STAGE_FRAGMENT_BIT) si.pSpecializationInfo = &ps_spec_info_;
            }
        }

        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = static_cast<std::uint32_t>(vertex_bindings_.size());
        vi.pVertexBindingDescriptions = vertex_bindings_.data();
        vi.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(vertex_attributes_.size());
        vi.pVertexAttributeDescriptions = vertex_attributes_.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = static_cast<VkPrimitiveTopology>(json::u32(p, "topology"));
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = static_cast<VkPolygonMode>(json::u32(p, "polygon_mode"));
        rs.cullMode = json::u32(p, "cull_mode");
        rs.frontFace = static_cast<VkFrontFace>(json::u32(p, "front_face"));
        rs.lineWidth = 1.0f;
        rs.depthClampEnable = json::flag(p, "depth_clamp") ? VK_TRUE : VK_FALSE;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = json::flag(p, "depth_test") ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = json::flag(p, "depth_write") ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = static_cast<VkCompareOp>(json::u32(p, "depth_compare"));
        ds.stencilTestEnable = json::flag(p, "stencil_test") ? VK_TRUE : VK_FALSE;
        ds.depthBoundsTestEnable = json::flag(p, "depth_bounds_test") ? VK_TRUE : VK_FALSE;
        if (ds.depthBoundsTestEnable && !dev_.depth_bounds) fail("the draw uses the depth-bounds test, which this device lacks");
        const auto stencil_state = [](const json::Value& s) {
            VkStencilOpState o{};
            o.failOp = static_cast<VkStencilOp>(json::u32(s, "fail_op"));
            o.passOp = static_cast<VkStencilOp>(json::u32(s, "pass_op"));
            o.depthFailOp = static_cast<VkStencilOp>(json::u32(s, "depth_fail_op"));
            o.compareOp = static_cast<VkCompareOp>(json::u32(s, "compare_op"));
            return o;
        };
        ds.front = stencil_state(json::member(p, "stencil_front"));
        ds.back = stencil_state(json::member(p, "stencil_back"));
        std::vector<VkPipelineColorBlendAttachmentState> blends;
        std::vector<VkFormat> formats;
        const auto& atts = json::arr(p, "attachments");
        if (atts.size() != colors_.size()) fail("pipeline attachments and colour targets disagree");
        for (std::size_t a = 0; a < atts.size(); ++a) {
            const json::Value& at = atts[a];
            if (integer(json::member(at, "slot"), "slot") != colors_[a].slot || json::u32(at, "format") != static_cast<std::uint32_t>(colors_[a].format)) {
                fail("attachment " + std::to_string(a) + " does not match its colour target");
            }
            VkPipelineColorBlendAttachmentState b{};
            b.blendEnable = json::flag(at, "blend_enable") ? VK_TRUE : VK_FALSE;
            b.srcColorBlendFactor = static_cast<VkBlendFactor>(json::u32(at, "src_color"));
            b.dstColorBlendFactor = static_cast<VkBlendFactor>(json::u32(at, "dst_color"));
            b.colorBlendOp = static_cast<VkBlendOp>(json::u32(at, "color_op"));
            b.srcAlphaBlendFactor = static_cast<VkBlendFactor>(json::u32(at, "src_alpha"));
            b.dstAlphaBlendFactor = static_cast<VkBlendFactor>(json::u32(at, "dst_alpha"));
            b.alphaBlendOp = static_cast<VkBlendOp>(json::u32(at, "alpha_op"));
            b.colorWriteMask = json::u32(at, "write_mask");
            blends.push_back(b);
            formats.push_back(colors_[a].format);
        }
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = static_cast<std::uint32_t>(blends.size());
        cb.pAttachments = blends.data();
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                           VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                           VK_DYNAMIC_STATE_BLEND_CONSTANTS};
        if (!is_null(json::member(m, "dynamic"), "depth_bounds")) {
            if (!dev_.depth_bounds) fail("the capture sets depth bounds, which this device lacks");
            dyn.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
        }
        VkPipelineDynamicStateCreateInfo dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dsi.dynamicStateCount = static_cast<std::uint32_t>(dyn.size());
        dsi.pDynamicStates = dyn.data();
        VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        ri.colorAttachmentCount = static_cast<std::uint32_t>(formats.size());
        ri.pColorAttachmentFormats = formats.data();
        ri.depthAttachmentFormat = static_cast<VkFormat>(json::u32(p, "depth_format"));
        ri.stencilAttachmentFormat = static_cast<VkFormat>(json::u32(p, "stencil_format"));
        VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpci.pNext = &ri;
        gpci.stageCount = static_cast<std::uint32_t>(infos.size());
        gpci.pStages = infos.data();
        gpci.pVertexInputState = &vi;
        gpci.pInputAssemblyState = &ia;
        gpci.pViewportState = &vp;
        gpci.pRasterizationState = &rs;
        gpci.pMultisampleState = &ms;
        gpci.pDepthStencilState = &ds;
        gpci.pColorBlendState = &cb;
        gpci.pDynamicState = &dsi;
        gpci.layout = layout_;
        VkPipeline pipeline = VK_NULL_HANDLE;
        if (!library) {
            check(vkCreateGraphicsPipelines(dev_.device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline), "vkCreateGraphicsPipelines");
            pipelines_.push_back(pipeline);
            return pipeline;
        }
        // --library: bbhost's BBHOST_PIPELINE_LIBRARY split (src/host/render.cpp
        // link_gfx_pipeline), fast-linked from four libraries that all use the
        // pipeline's full layout, with cull, front face, depth and stencil state
        // dynamic. (Partial layouts with independent sets lost the device.)
        if (!dev_.gpl) fail("--library: the device lacks VK_EXT_graphics_pipeline_library");
        const bool lto = std::getenv("DRAWREPLAY_LIB_LTO") != nullptr;  // diagnostics: link with link-time optimization
        const auto make = [&](VkGraphicsPipelineCreateInfo& info, VkGraphicsPipelineLibraryFlagsEXT flags, void* next) {
            VkGraphicsPipelineLibraryCreateInfoEXT li{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT};
            li.flags = flags;
            li.pNext = next;
            info.pNext = &li;
            info.flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | (lto ? VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT : 0);
            VkPipeline lib = VK_NULL_HANDLE;
            check(vkCreateGraphicsPipelines(dev_.device, VK_NULL_HANDLE, 1, &info, nullptr, &lib), "vkCreateGraphicsPipelines (library)");
            pipelines_.push_back(lib);
            return lib;
        };
        VkGraphicsPipelineCreateInfo vii{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        vii.pVertexInputState = &vi;
        vii.pInputAssemblyState = &ia;
        const VkPipeline l_vi = make(vii, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, nullptr);
        std::vector<VkPipelineShaderStageCreateInfo> pre, frag;
        for (const VkPipelineShaderStageCreateInfo& si : infos) (si.stage == VK_SHADER_STAGE_FRAGMENT_BIT ? frag : pre).push_back(si);
        const VkDynamicState pr_dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE};
        VkPipelineDynamicStateCreateInfo pr_dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        pr_dsi.dynamicStateCount = 4;
        pr_dsi.pDynamicStates = pr_dyn;
        VkPipelineRasterizationStateCreateInfo prs = rs;
        prs.cullMode = VK_CULL_MODE_NONE;
        prs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkGraphicsPipelineCreateInfo pri{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pri.stageCount = static_cast<std::uint32_t>(pre.size());
        pri.pStages = pre.data();
        pri.pViewportState = &vp;
        pri.pRasterizationState = &prs;
        pri.pDynamicState = &pr_dsi;
        pri.layout = layout_;
        // Like the fragment shader, pre-rasterization takes no target formats:
        // bbhost compiles vertex shaders before any draw names its targets.
        VkPipelineRenderingCreateInfo pr_no_formats{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        const VkPipeline l_pr = make(pri, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, &pr_no_formats);
        std::vector<VkDynamicState> fs_dyn = {VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,    VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                                              VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,  VK_DYNAMIC_STATE_STENCIL_OP,         VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                              VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK};
        if (dev_.depth_bounds) {
            fs_dyn.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
            fs_dyn.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
        }
        VkPipelineDynamicStateCreateInfo fs_dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        fs_dsi.dynamicStateCount = static_cast<std::uint32_t>(fs_dyn.size());
        fs_dsi.pDynamicStates = fs_dyn.data();
        VkPipelineDepthStencilStateCreateInfo fds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkGraphicsPipelineCreateInfo fsi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        fsi.stageCount = static_cast<std::uint32_t>(frag.size());
        fsi.pStages = frag.data();
        fsi.pMultisampleState = &ms;
        fsi.pDepthStencilState = &fds;
        fsi.pDynamicState = &fs_dsi;
        fsi.layout = layout_;
        // No target formats: bbhost compiles the fragment-shader library before a draw knows them.
        VkPipelineRenderingCreateInfo fs_no_formats{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        const VkPipeline l_fs = make(fsi, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, &fs_no_formats);
        const VkDynamicState fo_dyn[] = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};
        VkPipelineDynamicStateCreateInfo fo_dsi{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        fo_dsi.dynamicStateCount = 1;
        fo_dsi.pDynamicStates = fo_dyn;
        VkGraphicsPipelineCreateInfo foi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        foi.pColorBlendState = &cb;
        foi.pMultisampleState = &ms;
        foi.pDynamicState = &fo_dsi;
        const VkPipeline l_fo = make(foi, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, &ri);
        const VkPipeline libs[4] = {l_vi, l_pr, l_fs, l_fo};
        VkPipelineLibraryCreateInfoKHR lci{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
        lci.libraryCount = 4;
        lci.pLibraries = libs;
        VkGraphicsPipelineCreateInfo link{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        link.pNext = &lci;
        link.layout = layout_;
        if (lto) link.flags = VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
        check(vkCreateGraphicsPipelines(dev_.device, VK_NULL_HANDLE, 1, &link, nullptr, &pipeline), "vkCreateGraphicsPipelines (link)");
        pipelines_.push_back(pipeline);
        return pipeline;
    }

    RunResult run(const RunOptions& ro) {
        const json::Value& m = cap_.m;
        reset_inputs(ro.sentinel);
        // Targets from the captured initial contents.
        std::vector<Image> color_images;
        std::vector<VkImageView> color_views;
        for (const ColorTarget& t : colors_) {
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = t.format;
            ici.extent = {t.width, t.height, 1};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            color_images.push_back(dev_.image(ici));
            std::vector<std::uint8_t> initial = cap_.blob(t.initial);
            if (ro.perturb_initial && &t == &colors_.front()) perturb_texel(initial, t.format, replay::kAspectColor, t.width, ro.px, ro.py);
            upload(dev_, color_images.back(), VK_IMAGE_ASPECT_COLOR_BIT, initial, {whole(t.format, replay::kAspectColor, t.width, t.height)},
                   VK_IMAGE_LAYOUT_UNDEFINED);
            color_views.push_back(make_view(color_images.back().image, t.format, VK_IMAGE_ASPECT_COLOR_BIT));
        }
        Image depth_image;
        VkImageView depth_view = VK_NULL_HANDLE;
        std::vector<VkImageView> run_views;
        if (has_depth_) {
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = depth_format_;
            ici.extent = {depth_w_, depth_h_, 1};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            depth_image = dev_.image(ici);
            upload(dev_, depth_image, VK_IMAGE_ASPECT_DEPTH_BIT, cap_.blob(depth_initial_), {whole(depth_format_, replay::kAspectDepth, depth_w_, depth_h_)},
                   VK_IMAGE_LAYOUT_UNDEFINED);
            if (has_stencil_) {
                upload(dev_, depth_image, VK_IMAGE_ASPECT_STENCIL_BIT, cap_.blob(stencil_initial_),
                       {whole(depth_format_, replay::kAspectStencil, depth_w_, depth_h_)}, VK_IMAGE_LAYOUT_GENERAL);
            }
            depth_view = make_view(depth_image.image, depth_format_, full_aspect(depth_format_));
        }

        // Descriptor sets; views that alias the depth target view this run's image.
        check(vkResetDescriptorPool(dev_.device, pool_, 0), "vkResetDescriptorPool");
        VkDescriptorSet sets[3] = {};
        VkDescriptorSetLayout layouts[3] = {set_layout_, set_layout_, cb_set_layout_};
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = pool_;
        dsai.descriptorSetCount = 3;
        dsai.pSetLayouts = layouts;
        check(vkAllocateDescriptorSets(dev_.device, &dsai, sets), "vkAllocateDescriptorSets");
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorImageInfo> image_infos;
        std::vector<VkDescriptorBufferInfo> buffer_infos;
        image_infos.reserve(256);
        buffer_infos.reserve(32);
        const auto& stages = json::arr(m, "stages");
        for (int st = 0; st < 2; ++st) {
            const StageData& sd = stages_[st];
            buffer_infos.push_back({sd.params.buffer, 0, sizeof(gcn::StageParams)});
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = sets[st];
            w.dstBinding = gcn::kBindingParams;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.pBufferInfo = &buffer_infos.back();
            writes.push_back(w);
            if (!json::flag(stages[st], "present")) continue;
            for (const json::Value& b : json::arr(stages[st], "images")) {
                const std::uint32_t binding = json::u32(b, "binding");
                if (binding < gcn::kBindingImage0 || binding >= gcn::kBindingImage0 + 32) fail("image binding out of range");
                const std::string& id = json::str(b, "image");
                VkImageView view = VK_NULL_HANDLE;
                const json::Value& res = json::member(json::member(m, "images"), id.c_str());
                if (!is_null(res, "alias")) {
                    if (json::str(res, "alias") != "depth" || !has_depth_) fail("image " + id + " aliases an unknown target");
                    VkImageViewCreateInfo vci = view_info_from(json::member(res, "view"), depth_image.image);
                    check(vkCreateImageView(dev_.device, &vci, nullptr, &view), "vkCreateImageView (depth alias)");
                    run_views.push_back(view);
                } else {
                    view = views_.at(id);
                }
                image_infos.push_back({VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL});
                VkWriteDescriptorSet wi{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                wi.dstSet = sets[st];
                wi.dstBinding = binding;
                wi.descriptorCount = 1;
                wi.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                wi.pImageInfo = &image_infos.back();
                writes.push_back(wi);
            }
            for (const json::Value& b : json::arr(stages[st], "samplers")) {
                const std::uint32_t binding = json::u32(b, "binding");
                if (binding < gcn::kBindingSampler0 || binding >= gcn::kBindingSampler0 + 32) fail("sampler binding out of range");
                image_infos.push_back({samplers_.at(json::str(b, "sampler")), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                ws.dstSet = sets[st];
                ws.dstBinding = binding;
                ws.descriptorCount = 1;
                ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                ws.pImageInfo = &image_infos.back();
                writes.push_back(ws);
            }
            const auto& buffers = json::arr(stages[st], "buffers");
            for (std::size_t i = 0; i < buffers.size(); ++i) {
                const std::uint32_t binding = json::u32(buffers[i], "binding");
                if (binding < gcn::kBindingStorageBuffer0 || binding >= gcn::kBindingStorageBuffer0 + 2 * gcn::kMaxBuffers) fail("buffer binding out of range");
                const bool in_set2 = spirv_uses_set(words_of(cap_.blob(json::str(stages[st], "spirv")), "stage SPIR-V"), 2);
                if (!in_set2 && binding >= gcn::kBindingStorageBuffer0 + gcn::kMaxBuffers) fail("buffer binding out of range");
                buffer_infos.push_back({sd.buffers[i].buffer, 0, sd.buffer_bytes[i]});
                VkWriteDescriptorSet wb{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                wb.dstSet = in_set2 ? sets[2] : sets[st];
                wb.dstBinding = binding;
                wb.descriptorCount = 1;
                wb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wb.pBufferInfo = &buffer_infos.back();
                writes.push_back(wb);
            }
        }
        vkUpdateDescriptorSets(dev_.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // Readback staging for every target plane.
        struct Plane {
            Output out;
            VkImage image;
            Buffer staging;
        };
        std::vector<Plane> planes;
        for (std::size_t a = 0; a < colors_.size(); ++a) {
            Plane p;
            p.out = {"color" + std::to_string(colors_[a].slot), colors_[a].format, replay::kAspectColor, colors_[a].width, colors_[a].height, {}};
            p.image = color_images[a].image;
            planes.push_back(p);
        }
        if (has_depth_) {
            Plane p;
            p.out = {"depth", depth_format_, replay::kAspectDepth, depth_w_, depth_h_, {}};
            p.image = depth_image.image;
            planes.push_back(p);
            if (has_stencil_) {
                p.out = {"stencil", depth_format_, replay::kAspectStencil, depth_w_, depth_h_, {}};
                planes.push_back(p);
            }
        }
        for (Plane& p : planes) {
            p.staging = dev_.buffer(replay::subresource_bytes(p.out.format, p.out.aspect, p.out.width, p.out.height, 1),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }

        const json::Value& dyn = json::member(m, "dynamic");
        const json::Value& draw = json::member(m, "draw");
        VkQueryPool qpool = VK_NULL_HANDLE;
        if (ro.timed) {
            if (!dev_.timestamp_period_ns) fail("timing needs timestamp queries, which this queue lacks");
            VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qci.queryCount = 2;
            check(vkCreateQueryPool(dev_.device, &qci, nullptr, &qpool), "vkCreateQueryPool");
        }
        dev_.submit([&](VkCommandBuffer cmd) {
            if (qpool) vkCmdResetQueryPool(cmd, qpool, 0, 2);
            std::vector<VkRenderingAttachmentInfo> atts;
            for (VkImageView v : color_views) {
                VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
                a.imageView = v;
                a.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                a.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                atts.push_back(a);
            }
            VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            depth.imageView = depth_view;
            depth.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            const auto& area = json::arr(dyn, "render_area");
            if (area.size() != 2) fail("render_area must have two members");
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {u32_of(area[0], "render_area"), u32_of(area[1], "render_area")}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = static_cast<std::uint32_t>(atts.size());
            ri.pColorAttachments = atts.data();
            if (has_depth_) ri.pDepthAttachment = &depth;
            if (has_stencil_) ri.pStencilAttachment = &depth;
            vkCmdBeginRendering(cmd, &ri);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ro.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 3, sets, 0, nullptr);
            if (ro.library) {  // the state bbhost sets per draw for a pipeline linked from libraries
                const json::Value& pj = json::member(m, "pipeline");
                const auto ops = [&](VkStencilFaceFlags face, const json::Value& s) {
                    vkCmdSetStencilOp(cmd, face, static_cast<VkStencilOp>(json::u32(s, "fail_op")), static_cast<VkStencilOp>(json::u32(s, "pass_op")),
                                      static_cast<VkStencilOp>(json::u32(s, "depth_fail_op")), static_cast<VkCompareOp>(json::u32(s, "compare_op")));
                };
                vkCmdSetCullMode(cmd, json::u32(pj, "cull_mode"));
                vkCmdSetFrontFace(cmd, static_cast<VkFrontFace>(json::u32(pj, "front_face")));
                vkCmdSetDepthTestEnable(cmd, json::flag(pj, "depth_test") ? VK_TRUE : VK_FALSE);
                vkCmdSetDepthWriteEnable(cmd, json::flag(pj, "depth_write") ? VK_TRUE : VK_FALSE);
                vkCmdSetDepthCompareOp(cmd, static_cast<VkCompareOp>(json::u32(pj, "depth_compare")));
                vkCmdSetStencilTestEnable(cmd, json::flag(pj, "stencil_test") ? VK_TRUE : VK_FALSE);
                ops(VK_STENCIL_FACE_FRONT_BIT, json::member(pj, "stencil_front"));
                ops(VK_STENCIL_FACE_BACK_BIT, json::member(pj, "stencil_back"));
                if (dev_.depth_bounds) {
                    vkCmdSetDepthBoundsTestEnable(cmd, json::flag(pj, "depth_bounds_test") ? VK_TRUE : VK_FALSE);
                    if (is_null(dyn, "depth_bounds")) vkCmdSetDepthBounds(cmd, 0.0f, 1.0f);
                }
            }
            const json::Value& vpj = json::member(dyn, "viewport");
            VkViewport viewport{json::read_f32(vpj, "x"), json::read_f32(vpj, "y"), json::read_f32(vpj, "width"),
                                json::read_f32(vpj, "height"), json::read_f32(vpj, "min_depth"), json::read_f32(vpj, "max_depth")};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            const auto& sc = json::arr(dyn, "scissor");
            if (sc.size() != 4) fail("scissor must have four members");
            VkRect2D scissor{{static_cast<std::int32_t>(integer(sc[0], "scissor")), static_cast<std::int32_t>(integer(sc[1], "scissor"))},
                             {u32_of(sc[2], "scissor"), u32_of(sc[3], "scissor")}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            const json::Value& stencil = json::member(dyn, "stencil");
            vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_BIT, json::u32(stencil, "reference_front"));
            vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_BACK_BIT, json::u32(stencil, "reference_back"));
            vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_BIT, json::u32(stencil, "compare_mask_front"));
            vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_BACK_BIT, json::u32(stencil, "compare_mask_back"));
            vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_BIT, json::u32(stencil, "write_mask_front"));
            vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_BACK_BIT, json::u32(stencil, "write_mask_back"));
            const auto& bc = json::arr(dyn, "blend_constants");
            if (bc.size() != 4) fail("blend_constants must have four members");
            const float blend[4] = {json::as_f32(bc[0], "blend_constants"), json::as_f32(bc[1], "blend_constants"),
                                    json::as_f32(bc[2], "blend_constants"), json::as_f32(bc[3], "blend_constants")};
            vkCmdSetBlendConstants(cmd, blend);
            if (!is_null(dyn, "depth_bounds")) {
                const auto& bounds = json::arr(dyn, "depth_bounds");
                if (bounds.size() != 2) fail("depth_bounds must have two members");
                vkCmdSetDepthBounds(cmd, json::as_f32(bounds[0], "depth_bounds"), json::as_f32(bounds[1], "depth_bounds"));
            }
            const std::uint32_t count = json::u32(draw, "vertex_count"), instances = json::u32(draw, "instance_count");
            const std::int32_t base_vertex = static_cast<std::int32_t>(integer(json::member(draw, "base_vertex"), "base_vertex"));
            const bool indexed = json::flag(draw, "indexed");
            if (!vertex_buffers_.empty()) {
                std::vector<VkBuffer> vbs;
                for (const Buffer& b : vertex_buffers_) vbs.push_back(b.buffer);
                const std::vector<VkDeviceSize> offsets(vbs.size(), 0);
                vkCmdBindVertexBuffers(cmd, 0, static_cast<std::uint32_t>(vbs.size()), vbs.data(), offsets.data());
            }
            if (indexed) {
                const VkIndexType type = json::str(draw, "index_type") == "uint32" ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
                if (index_.size < static_cast<VkDeviceSize>(count) * (type == VK_INDEX_TYPE_UINT32 ? 4 : 2)) fail("index blob shorter than the draw");
                vkCmdBindIndexBuffer(cmd, index_.buffer, 0, type);
            }
            if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
            for (std::uint32_t k = 0; k < std::max(ro.repeat, 1u); ++k) {
                if (indexed) vkCmdDrawIndexed(cmd, count, instances, 0, base_vertex, 0);
                else vkCmdDraw(cmd, count, instances, static_cast<std::uint32_t>(base_vertex), 0);
            }
            if (qpool) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
            vkCmdEndRendering(cmd);
            for (Plane& p : planes) {
                VkBufferImageCopy r{};
                r.imageSubresource = {p.out.aspect, 0, 0, 1};
                r.imageExtent = {p.out.width, p.out.height, 1};
                vkCmdCopyImageToBuffer(cmd, p.image, VK_IMAGE_LAYOUT_GENERAL, p.staging.buffer, 1, &r);
            }
        });

        RunResult result;
        result.name = ro.name;
        if (qpool) {
            std::uint64_t ts[2] = {};
            check(vkGetQueryPoolResults(dev_.device, qpool, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                  "vkGetQueryPoolResults");
            vkDestroyQueryPool(dev_.device, qpool, nullptr);
            result.gpu_ms = ts[1] > ts[0] ? static_cast<double>(ts[1] - ts[0]) * dev_.timestamp_period_ns / 1e6 : 0.0;
        }
        for (Plane& p : planes) {
            const std::size_t bytes = static_cast<std::size_t>(replay::subresource_bytes(p.out.format, p.out.aspect, p.out.width, p.out.height, 1));
            p.out.bytes.assign(static_cast<const std::uint8_t*>(p.staging.map), static_cast<const std::uint8_t*>(p.staging.map) + bytes);
            dev_.destroy(p.staging);
            result.outputs.push_back(std::move(p.out));
        }
        result.writes = changed_inputs(ro.sentinel);
        result.stored = stored_buffers();
        for (VkImageView v : run_views) vkDestroyImageView(dev_.device, v, nullptr);
        for (VkImageView v : color_views) vkDestroyImageView(dev_.device, v, nullptr);
        if (depth_view) vkDestroyImageView(dev_.device, depth_view, nullptr);
        for (Image& img : color_images) dev_.destroy(img);
        dev_.destroy(depth_image);
        return result;
    }

    // The game's own output of the draw, laid out like a run's outputs.
    std::vector<Output> live() const {
        std::vector<Output> out;
        for (const ColorTarget& t : colors_) {
            out.push_back({"color" + std::to_string(t.slot), t.format, replay::kAspectColor, t.width, t.height, cap_.blob(t.live)});
        }
        if (has_depth_) {
            out.push_back({"depth", depth_format_, replay::kAspectDepth, depth_w_, depth_h_, cap_.blob(depth_live_)});
            if (has_stencil_) out.push_back({"stencil", depth_format_, replay::kAspectStencil, depth_w_, depth_h_, cap_.blob(stencil_live_)});
        }
        return out;
    }

    static void perturb_texel(std::vector<std::uint8_t>& bytes, VkFormat format, std::uint32_t aspect, std::uint32_t width, std::uint32_t x,
                              std::uint32_t y) {
        std::uint32_t texel = 0, bw = 1, bh = 1;
        if (!replay::texel_layout(format, aspect, texel, bw, bh)) fail("cannot perturb an unknown texel layout");
        const std::size_t at = (static_cast<std::size_t>(y) * width + x) * texel + (texel > 1 ? 1 : 0);
        if (at >= bytes.size()) fail("perturbed texel outside the plane");
        bytes[at] ^= 0x40;  // an exponent or high-order bit in every layout the targets use
    }

private:
    struct StageData {
        Buffer params;
        std::vector<Buffer> buffers;
        std::vector<VkDeviceSize> buffer_bytes;
        std::vector<std::string> buffer_blobs;
        std::vector<bool> buffer_written;  // the shaders store to it (an output, not an input)
    };

    static Subresource whole(VkFormat format, std::uint32_t aspect, std::uint32_t w, std::uint32_t h) {
        Subresource s;
        s.w = w;
        s.h = h;
        s.bytes = replay::subresource_bytes(format, aspect, w, h, 1);
        return s;
    }

    VkImageView make_view(VkImage image, VkFormat format, VkImageAspectFlags aspect) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = {aspect, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        check(vkCreateImageView(dev_.device, &vci, nullptr, &view), "vkCreateImageView (target)");
        return view;
    }

    void load_targets() {
        const json::Value& targets = json::member(cap_.m, "targets");
        for (const json::Value& t : json::arr(targets, "color")) {
            ColorTarget c;
            c.slot = static_cast<int>(integer(json::member(t, "slot"), "slot"));
            c.format = static_cast<VkFormat>(json::u32(t, "format"));
            const auto& ext = json::arr(t, "extent");
            if (ext.size() != 2) fail("target extent must have two members");
            c.width = u32_of(ext[0], "extent");
            c.height = u32_of(ext[1], "extent");
            c.initial = json::str(t, "initial");
            c.live = json::str(t, "live");
            for (const std::string* b : {&c.initial, &c.live}) {
                if (cap_.blob(*b).size() != replay::subresource_bytes(c.format, replay::kAspectColor, c.width, c.height, 1)) {
                    fail("colour target blob " + *b + " does not match its format and extent");
                }
            }
            colors_.push_back(c);
        }
        if (is_null(targets, "depth")) return;
        const json::Value& d = json::member(targets, "depth");
        has_depth_ = true;
        depth_format_ = static_cast<VkFormat>(json::u32(d, "format"));
        const auto& ext = json::arr(d, "extent");
        if (ext.size() != 2) fail("depth extent must have two members");
        depth_w_ = u32_of(ext[0], "extent");
        depth_h_ = u32_of(ext[1], "extent");
        depth_initial_ = json::str(d, "depth_initial");
        depth_live_ = json::str(d, "depth_live");
        has_stencil_ = json::u32(json::member(cap_.m, "pipeline"), "stencil_format") != VK_FORMAT_UNDEFINED;
        if (has_stencil_) {
            stencil_initial_ = json::str(d, "stencil_initial");
            stencil_live_ = json::str(d, "stencil_live");
        }
    }

    void build_memory() {
        const json::Value& memory = json::member(cap_.m, "memory");
        if (json::u64(memory, "page_bytes") != kPage) fail("capture page size differs from the translator's");
        const auto& blob = cap_.blob(json::str(memory, "blob"));
        for (const json::Value& p : json::arr(memory, "pages")) {
            const std::uint64_t va = json::as_u64(p, "pages");
            if (va & (kPage - 1) || (va >> 32) >= gcn::kL1Entries) fail("guest page address out of range");
            page_vas_.push_back(va);
        }
        if (blob.size() != page_vas_.size() * kPage) fail("memory blob does not hold one 64 KiB block per page");
        pages_ = dev_.buffer(std::max<std::uint64_t>(blob.size(), kPage), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        sink_ = dev_.buffer(kPage, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        sink_l2_ = dev_.buffer(gcn::kL2Entries * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        l1_ = dev_.buffer(gcn::kL1Entries * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        const auto fill_l2 = [&](Buffer& t) {
            auto* e = static_cast<std::uint64_t*>(t.map);
            for (std::uint32_t k = 0; k < gcn::kL2Entries; ++k) e[k] = sink_.address;
        };
        fill_l2(sink_l2_);
        for (std::uint32_t k = 0; k < gcn::kL1Entries; ++k) static_cast<std::uint64_t*>(l1_.map)[k] = sink_l2_.address;
        for (std::size_t i = 0; i < page_vas_.size(); ++i) {
            const std::uint32_t i1 = static_cast<std::uint32_t>(page_vas_[i] >> 32);
            auto it = l2_.find(i1);
            if (it == l2_.end()) {
                Buffer t = dev_.buffer(gcn::kL2Entries * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                fill_l2(t);
                static_cast<std::uint64_t*>(l1_.map)[i1] = t.address;
                it = l2_.emplace(i1, t).first;
            }
            static_cast<std::uint64_t*>(it->second.map)[(page_vas_[i] >> gcn::kPageShift) & (gcn::kL2Entries - 1)] = pages_.address + i * kPage;
        }
    }

    void build_layout() {
        // bbhost's set layout (src/host/gpu.cpp): the modules were translated against it.
        std::vector<VkDescriptorSetLayoutBinding> binds;
        binds.push_back({gcn::kBindingParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr});
        for (std::uint32_t k = 0; k < 32; ++k) {
            binds.push_back({gcn::kBindingImage0 + k, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr});
            binds.push_back({gcn::kBindingSampler0 + k, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_ALL, nullptr});
            binds.push_back({gcn::kBindingStorageImage0 + k, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL, nullptr});
        }
        for (std::uint32_t k = 0; k < gcn::kMaxBuffers; ++k) {
            binds.push_back({gcn::kBindingStorageBuffer0 + k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr});
        }
        VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        lci.bindingCount = static_cast<std::uint32_t>(binds.size());
        lci.pBindings = binds.data();
        check(vkCreateDescriptorSetLayout(dev_.device, &lci, nullptr, &set_layout_), "vkCreateDescriptorSetLayout");
        // Set 2: the constant buffers, both stages' bindings (spirv_uses_set).
        std::vector<VkDescriptorSetLayoutBinding> cbs;
        for (std::uint32_t k = 0; k < 2 * gcn::kMaxBuffers; ++k) {
            cbs.push_back({gcn::kBindingStorageBuffer0 + k, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr});
        }
        VkDescriptorSetLayoutCreateInfo cci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        cci.bindingCount = static_cast<std::uint32_t>(cbs.size());
        cci.pBindings = cbs.data();
        check(vkCreateDescriptorSetLayout(dev_.device, &cci, nullptr, &cb_set_layout_), "vkCreateDescriptorSetLayout");
        VkDescriptorSetLayout three[3] = {set_layout_, set_layout_, cb_set_layout_};
        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 3;
        plci.pSetLayouts = three;
        check(vkCreatePipelineLayout(dev_.device, &plci, nullptr, &layout_), "vkCreatePipelineLayout");
        const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2},  {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64}, {VK_DESCRIPTOR_TYPE_SAMPLER, 64},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * gcn::kMaxBuffers},
        };
        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 3;
        dpci.poolSizeCount = 5;
        dpci.pPoolSizes = sizes;
        check(vkCreateDescriptorPool(dev_.device, &dpci, nullptr, &pool_), "vkCreateDescriptorPool");
    }

    void build_resources() {
        for (const auto& [id, res] : json::member(cap_.m, "images").object) {
            if (!is_null(res, "alias")) continue;
            const VkImageCreateInfo ici = image_info_from(json::member(res, "image"));
            Image img = dev_.image(ici);
            images_[id] = img;
            const json::Value& view = json::member(res, "view");
            upload(dev_, img, json::u32(view, "aspect"), cap_.blob(json::str(res, "blob")), subresources_from(json::member(res, "subresources")),
                   VK_IMAGE_LAYOUT_UNDEFINED);
            VkImageViewCreateInfo vci = view_info_from(view, img.image);
            VkImageView v = VK_NULL_HANDLE;
            check(vkCreateImageView(dev_.device, &vci, nullptr, &v), ("vkCreateImageView " + id).c_str());
            views_[id] = v;
        }
        for (const auto& [id, s] : json::member(cap_.m, "samplers").object) {
            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            sci.flags = json::u32(s, "flags");
            sci.magFilter = static_cast<VkFilter>(json::u32(s, "mag_filter"));
            sci.minFilter = static_cast<VkFilter>(json::u32(s, "min_filter"));
            sci.mipmapMode = static_cast<VkSamplerMipmapMode>(json::u32(s, "mipmap_mode"));
            sci.addressModeU = static_cast<VkSamplerAddressMode>(json::u32(s, "address_u"));
            sci.addressModeV = static_cast<VkSamplerAddressMode>(json::u32(s, "address_v"));
            sci.addressModeW = static_cast<VkSamplerAddressMode>(json::u32(s, "address_w"));
            sci.mipLodBias = json::read_f32(s, "mip_lod_bias");
            sci.anisotropyEnable = json::flag(s, "anisotropy_enable") ? VK_TRUE : VK_FALSE;
            sci.maxAnisotropy = json::read_f32(s, "max_anisotropy");
            sci.compareEnable = json::flag(s, "compare_enable") ? VK_TRUE : VK_FALSE;
            sci.compareOp = static_cast<VkCompareOp>(json::u32(s, "compare_op"));
            sci.minLod = json::read_f32(s, "min_lod");
            sci.maxLod = json::read_f32(s, "max_lod");
            sci.borderColor = static_cast<VkBorderColor>(json::u32(s, "border_color"));
            sci.unnormalizedCoordinates = json::flag(s, "unnormalized_coordinates") ? VK_TRUE : VK_FALSE;
            VkSampler smp = VK_NULL_HANDLE;
            check(vkCreateSampler(dev_.device, &sci, nullptr, &smp), ("vkCreateSampler " + id).c_str());
            samplers_[id] = smp;
        }
    }

    void build_stages() {
        const auto& stages = json::arr(cap_.m, "stages");
        if (stages.size() != 2) fail("a capture has a vertex and a pixel stage entry");
        std::vector<std::uint8_t> fallback_params;
        for (int st = 0; st < 2; ++st) {
            StageData& sd = stages_[st];
            sd.params = dev_.buffer((sizeof(gcn::StageParams) + 255) & ~std::size_t{255}, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            // A stage without a module still gets a set with a params block,
            // as in bbhost; nothing reads it.
            const bool present = json::flag(stages[st], "present");
            const json::Value& ref = present ? stages[st] : stages[0];
            const auto& params = cap_.blob(json::str(json::member(ref, "params"), "blob"));
            // Captures from before StageParams::vertex_formats hold the 288 bytes up to it.
            if (params.size() != sizeof(gcn::StageParams) && params.size() != offsetof(gcn::StageParams, vertex_formats)) {
                fail("stage params blob has the wrong size");
            }
            gcn::StageParams sp{};
            std::memcpy(&sp, params.data(), params.size());
            sp.l1_table = l1_.address;
            if (st == 0) {  // the conversions a vertex shader built with vertex_formats_from_params reads
                const json::Value* vi = cap_.m.find("vertex_input");
                if (vi && vi->type == json::Value::Type::Object) {
                    for (const json::Value& e : json::arr(*vi, "elements")) {
                        const std::uint32_t location = json::u32(e, "location");
                        if (location < 16) sp.vertex_formats[location] = gcn::vertex_format_descriptor(static_cast<std::uint32_t>(json::u64(e, "w3")));
                    }
                }
            }
            std::memcpy(sd.params.map, &sp, sizeof(sp));
            if (!present) continue;
            for (const json::Value& b : json::arr(stages[st], "buffers")) {
                const std::string& name = json::str(b, "blob");
                const auto& bytes = cap_.blob(name);
                if (bytes.size() != json::u64(b, "bytes") || bytes.empty()) fail("buffer blob " + name + " has the wrong size");
                Buffer buf = dev_.buffer((bytes.size() + 3) & ~std::size_t{3}, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                sd.buffers.push_back(buf);
                sd.buffer_bytes.push_back(bytes.size());
                sd.buffer_blobs.push_back(name);
                const json::Value* written = b.find("written");  // set only on a buffer the shaders store to
                sd.buffer_written.push_back(written && json::flag(b, "written"));
            }
        }
    }

    void reset_inputs(bool sentinel) {
        const auto& blob = cap_.blob(json::str(json::member(cap_.m, "memory"), "blob"));
        if (!blob.empty()) std::memcpy(pages_.map, blob.data(), blob.size());
        auto* sink = static_cast<std::uint32_t*>(sink_.map);
        for (std::uint64_t k = 0; k < kPage / 4; ++k) sink[k] = sentinel ? sentinel_word(k) : 0u;
        for (StageData& sd : stages_) {
            for (std::size_t i = 0; i < sd.buffers.size(); ++i) {
                const auto& bytes = cap_.blob(sd.buffer_blobs[i]);
                std::memcpy(sd.buffers[i].map, bytes.data(), bytes.size());
            }
        }
    }

    std::vector<std::string> changed_inputs(bool sentinel) const {
        std::vector<std::string> out;
        const auto& blob = cap_.blob(json::str(json::member(cap_.m, "memory"), "blob"));
        for (std::size_t i = 0; i < page_vas_.size(); ++i) {
            if (std::memcmp(static_cast<const std::uint8_t*>(pages_.map) + i * kPage, blob.data() + i * kPage, kPage) != 0) {
                out.push_back("guest page " + json::hex(page_vas_[i]));
            }
        }
        const auto* sink = static_cast<const std::uint32_t*>(sink_.map);
        for (std::uint64_t k = 0; k < kPage / 4; ++k) {
            if (sink[k] != (sentinel ? sentinel_word(k) : 0u)) {
                out.push_back("sink page (an unmapped guest address)");
                break;
            }
        }
        for (int st = 0; st < 2; ++st) {
            const StageData& sd = stages_[st];
            for (std::size_t i = 0; i < sd.buffers.size(); ++i) {
                if (sd.buffer_written[i]) continue;  // an output: stored_buffers()
                const auto& bytes = cap_.blob(sd.buffer_blobs[i]);
                if (std::memcmp(sd.buffers[i].map, bytes.data(), bytes.size()) != 0) out.push_back("storage buffer " + sd.buffer_blobs[i]);
            }
        }
        return out;
    }
    std::vector<std::vector<std::uint8_t>> stored_buffers() const {
        std::vector<std::vector<std::uint8_t>> out;
        for (int st = 0; st < 2; ++st) {
            const StageData& sd = stages_[st];
            for (std::size_t i = 0; i < sd.buffers.size(); ++i) {
                if (!sd.buffer_written[i]) continue;
                const auto* p = static_cast<const std::uint8_t*>(sd.buffers[i].map);
                out.emplace_back(p, p + sd.buffer_bytes[i]);
            }
        }
        return out;
    }

    const Capture& cap_;
    Device& dev_;
    std::vector<ColorTarget> colors_;
    bool has_depth_ = false, has_stencil_ = false;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    std::uint32_t depth_w_ = 0, depth_h_ = 0;
    std::string depth_initial_, depth_live_, stencil_initial_, stencil_live_;
    Buffer pages_, sink_, sink_l2_, l1_, index_;
    std::vector<Buffer> vertex_buffers_;  // a vertex-input draw's bindings, in binding order
    std::vector<VkVertexInputBindingDescription> vertex_bindings_;
    std::vector<VkVertexInputAttributeDescription> vertex_attributes_;
    std::map<std::uint32_t, Buffer> l2_;
    std::vector<std::uint64_t> page_vas_;
    std::map<std::string, Image> images_;
    std::map<std::string, VkImageView> views_;
    std::map<std::string, VkSampler> samplers_;
    StageData stages_[2];
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout cb_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkShaderModule> modules_;

public:
    bool specialize = false;  // --specialize

private:
    // The words relink_pipeline specializes with: the vertex stage's 16
    // vertex-format descriptors (gcn::kVertexFormatSpecId), 16 words 3
    // (kCbW3SpecId) and 16 strides (kCbStrideSpecId); the pixel stage's words
    // 3 and strides. A 0 keeps the default, which reads the params.
    std::uint32_t vs_spec_[48] = {}, ps_spec_[32] = {};
    VkSpecializationMapEntry vs_entries_[48] = {}, ps_entries_[32] = {};
    VkSpecializationInfo vs_spec_info_{}, ps_spec_info_{};
    void specialization() {
        const auto& stages = json::arr(cap_.m, "stages");
        for (int st = 0; st < 2; ++st) {
            if (!json::flag(stages[st], "present")) continue;
            const auto& params = cap_.blob(json::str(json::member(stages[st], "params"), "blob"));
            gcn::StageParams sp{};
            std::memcpy(&sp, params.data(), std::min(params.size(), sizeof(sp)));
            std::uint32_t* words = st ? ps_spec_ : vs_spec_ + 16;
            for (int i = 0; i < 16; ++i) {  // render.cpp queue_library_relink: the indexed loads' (word 3 and stride set)
                if (!(sp.cb_w3[i] | sp.cb_stride[i])) continue;
                words[i] = sp.cb_w3[i];
                words[16 + i] = sp.cb_stride[i];
            }
        }
        const json::Value* fp = json::member(cap_.m, "translate").find("vertex_formats_from_params");
        const json::Value* vi = cap_.m.find("vertex_input");
        if (fp && fp->type == json::Value::Type::Bool && fp->boolean && vi && vi->type == json::Value::Type::Object) {
            for (const json::Value& e : json::arr(*vi, "elements")) {
                const std::uint32_t location = json::u32(e, "location");
                if (location < 16) vs_spec_[location] = gcn::vertex_format_descriptor(static_cast<std::uint32_t>(json::u64(e, "w3")));
            }
        }
        for (std::uint32_t k = 0; k < 16; ++k) {
            vs_entries_[k] = {gcn::kVertexFormatSpecId + k, k * 4, 4};
            vs_entries_[16 + k] = {gcn::kCbW3SpecId + k, (16 + k) * 4, 4};
            vs_entries_[32 + k] = {gcn::kCbStrideSpecId + k, (32 + k) * 4, 4};
            ps_entries_[k] = {gcn::kCbW3SpecId + k, k * 4, 4};
            ps_entries_[16 + k] = {gcn::kCbStrideSpecId + k, (16 + k) * 4, 4};
        }
        vs_spec_info_ = {48, vs_entries_, sizeof(vs_spec_), vs_spec_};
        ps_spec_info_ = {32, ps_entries_, sizeof(ps_spec_), ps_spec_};
    }
    std::vector<VkPipeline> pipelines_;
};

// ---- comparison and report ---------------------------------------------------------

struct Verdict {
    bool match = true;        // within tolerance everywhere
    bool identical = true;    // byte for byte
    std::uint64_t texels_failed = 0, texels_different = 0;
    json::Value json = json::Value::make_object();
};

Verdict compare_runs(const std::vector<Output>& a, const std::vector<Output>& b, const replay::Tolerance& tol, const fs::path& out,
                     const std::string& label, bool heatmap_always) {
    Verdict v;
    if (a.size() != b.size()) fail(label + ": runs have different target lists");
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].target != b[i].target || a[i].format != b[i].format || a[i].aspect != b[i].aspect) fail(label + ": targets differ");
        std::string error;
        replay::Plane pa, pb;
        if (!replay::decode_plane(a[i].format, a[i].aspect, a[i].bytes.data(), a[i].bytes.size(), a[i].width, a[i].height, pa, error) ||
            !replay::decode_plane(b[i].format, b[i].aspect, b[i].bytes.data(), b[i].bytes.size(), b[i].width, b[i].height, pb, error)) {
            fail(label + " " + a[i].target + ": " + error);
        }
        const replay::Comparison c = replay::compare_planes(pa, pb, tol);
        json::Value tj = replay::to_json(c);
        const bool identical = a[i].bytes == b[i].bytes;
        tj.set("bytes_identical", identical);
        if (!identical || heatmap_always) {
            const std::string name = "heatmap-" + label + "-" + a[i].target + ".ppm";
            if (replay::write_heatmap((out / name).string(), c)) tj.set("heatmap", name);
        }
        v.match = v.match && c.match();
        v.identical = v.identical && identical;
        v.texels_failed += c.texels_failed;
        v.texels_different += c.texels_different;
        v.json.set(a[i].target, tj);
    }
    return v;
}

json::Value verdict_json(const Verdict& v, bool pass, const char* rule) {
    json::Value j = json::Value::make_object();
    j.set("pass", pass);
    j.set("rule", rule);
    j.set("within_tolerance", v.match);
    j.set("bytes_identical", v.identical);
    j.set("texels_failed", v.texels_failed);
    j.set("texels_different", v.texels_different);
    j.set("targets", v.json);
    return j;
}

void save_outputs(const RunResult& r, const fs::path& out) {
    for (const Output& o : r.outputs) write_file(out / (r.name + "-" + o.target + ".bin"), o.bytes.data(), o.bytes.size());
}

// The written buffers of two runs, byte for byte: "" when equal, else where they first part.
std::string stored_difference(const RunResult& x, const RunResult& y) {
    if (x.stored.size() != y.stored.size()) return "a different number of written buffers";
    for (std::size_t i = 0; i < x.stored.size(); ++i) {
        if (x.stored[i].size() != y.stored[i].size()) return "written buffer " + std::to_string(i) + " has another size";
        for (std::size_t k = 0; k < x.stored[i].size(); ++k) {
            if (x.stored[i][k] != y.stored[i][k]) return "written buffer " + std::to_string(i) + " differs at byte " + std::to_string(k);
        }
    }
    return "";
}
std::size_t stored_bytes(const RunResult& r) {
    std::size_t n = 0;
    for (const auto& b : r.stored) n += b.size();
    return n;
}

json::Value writes_json(const RunResult& r) {
    json::Value a = json::Value::make_array();
    for (const std::string& w : r.writes) a.push(w);
    return a;
}

struct Options {
    fs::path capture;
    fs::path out;
    replay::Tolerance tol;
    fs::path ps;
    fs::path vs;
    bool live = true;
    bool validate = false;
    std::uint32_t bench = 0;  // draws per timed batch
    bool library = false;     // the variant is linked from pipeline libraries (BBHOST_PIPELINE_LIBRARY)
    bool specialize = false;  // every run with the optimized relink's specialization constants
};

[[noreturn]] void usage() {
    std::fprintf(stderr, "usage: drawreplay <capture-dir> [--out DIR] [--atol X] [--rtol Y] [--ps FILE.spv] [--vs FILE.spv] [--library] [--specialize] [--no-live] [--validate] [--bench N]\n");
    std::exit(2);
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (a == "--out") o.out = value();
        else if (a == "--atol") o.tol.atol = std::strtod(value().c_str(), nullptr);
        else if (a == "--rtol") o.tol.rtol = std::strtod(value().c_str(), nullptr);
        else if (a == "--ps") o.ps = value();
        else if (a == "--vs") o.vs = value();
        else if (a == "--library") o.library = true;
        else if (a == "--specialize") o.specialize = true;
        else if (a == "--no-live") o.live = false;
        else if (a == "--validate") o.validate = true;
        else if (a == "--bench") o.bench = static_cast<std::uint32_t>(std::strtoul(value().c_str(), nullptr, 10));
        else if (!a.empty() && a[0] == '-') usage();
        else if (o.capture.empty()) o.capture = a;
        else usage();
    }
    if (o.capture.empty() || !(o.tol.atol >= 0) || !(o.tol.rtol >= 0)) usage();
    return o;
}

int replay_main(const Options& o) {
    const Capture cap = load_capture(o.capture);
    fs::path out = o.out;
    if (out.empty()) {
        for (int n = 1;; ++n) {
            out = o.capture / ("replay-" + std::to_string(n));
            if (!fs::exists(out)) break;
        }
    }
    if (fs::exists(out)) fail(out.string() + " already exists; replay results are never overwritten");
    fs::create_directories(out);

    const json::Value& identity = json::member(cap.m, "identity");
    Device dev;
    dev.init(json::str(identity, "device"), o.validate);
    std::printf("drawreplay: %s (%s, flip %llu) on %s\n", o.capture.string().c_str(), json::str(identity, "pipeline").c_str(),
                static_cast<unsigned long long>(json::u64(identity, "flip")), dev.name.c_str());
    Replayer r(cap, dev);
    r.specialize = o.specialize;
    const VkPipeline captured = r.pipeline(nullptr);

    json::Value report = json::Value::make_object();
    report.set("schema", "bbhost-draw-replay");
    report.set("version", 1);
    report.set("capture", o.capture.string());
    report.set("capture_identity", identity);
    report.set("replay_device", dev.name);
    json::Value tol = json::Value::make_object();
    tol.set("atol", o.tol.atol);
    tol.set("rtol", o.tol.rtol);
    report.set("tolerance", tol);
    report.set("specialized", o.specialize);
    json::Value comparisons = json::Value::make_object();
    json::Value runs = json::Value::make_object();
    const replay::Tolerance exact;

    RunOptions ro;
    ro.pipeline = captured;
    ro.name = "A";
    const RunResult a = r.run(ro);
    ro.name = "B";
    const RunResult b = r.run(ro);
    save_outputs(a, out);
    save_outputs(b, out);
    const Verdict ab = compare_runs(a.outputs, b.outputs, o.tol, out, "A-vs-B", false);
    const std::string ab_stores = stored_difference(a, b);
    const bool deterministic = ab.match && a.writes.empty() && b.writes.empty() && ab_stores.empty();
    {
        json::Value abj = verdict_json(ab, deterministic, "the captured modules twice from identical captured inputs");
        if (!a.stored.empty()) {
            abj.set("written_buffers", static_cast<std::uint64_t>(a.stored.size()));
            abj.set("written_bytes", static_cast<std::uint64_t>(stored_bytes(a)));
            abj.set("stores", ab_stores.empty() ? std::string("identical") : ab_stores);
        }
        comparisons.set("A_vs_B", abj);
    }
    runs.set("A", writes_json(a));
    runs.set("B", writes_json(b));

    bool live_ok = true;
    if (o.live) {
        const Verdict lv = compare_runs(a.outputs, r.live(), o.tol, out, "A-vs-live", true);
        live_ok = lv.match;
        comparisons.set("A_vs_live", verdict_json(lv, live_ok, "replay against the game's own output of the draw (capture fidelity)"));
    }

    ro.name = "sentinel";
    ro.sentinel = true;
    const RunResult sentinel = r.run(ro);
    ro.sentinel = false;
    const Verdict sv = compare_runs(a.outputs, sentinel.outputs, exact, out, "A-vs-sentinel", false);
    const bool tracked = sv.identical && sentinel.writes.empty();
    json::Value sj = verdict_json(sv, tracked,
                                  "sink page filled with varied values must not change any output; supplementary to A_vs_live, "
                                  "since a read that does not reach the output (clipped geometry) is not detected");
    sj.set("sink_fill", "varied floats in [-1, 1], one per dword (sentinel_word)");
    comparisons.set("A_vs_sentinel", sj);
    runs.set("sentinel", writes_json(sentinel));

    // Self-tests: perturb one texel of A's output, and of the first colour
    // target's initial contents, at the middle of the scissored area.
    bool self_ok = true;
    json::Value self = json::Value::make_object();
    if (!r.colors().empty() && !a.outputs.empty()) {
        const json::Value& dyn = json::member(cap.m, "dynamic");
        const auto& sc = json::arr(dyn, "scissor");
        const ColorTarget& t0 = r.colors().front();
        const std::int64_t sx = std::max<std::int64_t>(0, integer(sc[0], "scissor")), sy = std::max<std::int64_t>(0, integer(sc[1], "scissor"));
        const std::uint32_t px = static_cast<std::uint32_t>(std::min<std::int64_t>(t0.width - 1, sx + integer(sc[2], "scissor") / 2));
        const std::uint32_t py = static_cast<std::uint32_t>(std::min<std::int64_t>(t0.height - 1, sy + integer(sc[3], "scissor") / 2));
        const auto detected_at = [&](const Verdict& v, const std::string& target) {
            if (v.texels_different != 1 || v.texels_failed != 1) return false;
            const json::Value& fails = json::member(json::member(v.json, target.c_str()), "first_failures");
            return fails.array.size() >= 1 && json::u32(fails.array[0], "x") == px && json::u32(fails.array[0], "y") == py;
        };
        std::vector<Output> perturbed = a.outputs;
        Replayer::perturb_texel(perturbed[0].bytes, perturbed[0].format, perturbed[0].aspect, perturbed[0].width, px, py);
        const Verdict ov = compare_runs(a.outputs, perturbed, exact, out, "self-output", false);
        const bool out_ok = detected_at(ov, a.outputs[0].target);
        json::Value oj = verdict_json(ov, out_ok, "one flipped bit in A's first colour output must fail exactly that texel");
        self.set("output_perturbation", oj);

        const json::Value& att = json::arr(json::member(cap.m, "pipeline"), "attachments").front();
        const bool dst_kept = json::u32(att, "write_mask") == 0 ||
                              (json::flag(att, "blend_enable") && (json::u32(att, "dst_color") != VK_BLEND_FACTOR_ZERO ||
                                                                  json::u32(att, "dst_alpha") != VK_BLEND_FACTOR_ZERO));
        json::Value ij;
        bool in_ok = true;
        if (dst_kept) {
            ro.name = "perturbed";
            ro.perturb_initial = true;
            ro.px = px;
            ro.py = py;
            const RunResult c = r.run(ro);
            ro.perturb_initial = false;
            const Verdict iv = compare_runs(a.outputs, c.outputs, exact, out, "self-input", false);
            in_ok = detected_at(iv, a.outputs[0].target) && c.writes.empty();
            ij = verdict_json(iv, in_ok, "one flipped bit in the first colour target's initial contents must change exactly that output texel");
        } else {
            ij = json::Value::make_object();
            ij.set("pass", true);
            ij.set("skipped", "the first attachment's blend discards the destination where shaded");
        }
        self.set("input_perturbation", ij);
        json::Value at = json::Value::make_array();
        at.push(px);
        at.push(py);
        self.set("texel", at);
        self_ok = out_ok && in_ok;
    }
    comparisons.set("self_tests", self);

    VkPipeline variant_pipeline = VK_NULL_HANDLE;
    if (!o.ps.empty() || !o.vs.empty() || o.library) {
        const std::vector<std::uint32_t> ps = o.ps.empty() ? std::vector<std::uint32_t>() : words_of(read_file(o.ps), o.ps.string());
        const std::vector<std::uint32_t> vs = o.vs.empty() ? std::vector<std::uint32_t>() : words_of(read_file(o.vs), o.vs.string());
        ro.name = "variant";
        ro.pipeline = variant_pipeline = r.pipeline(o.ps.empty() ? nullptr : &ps, o.vs.empty() ? nullptr : &vs, o.library);
        ro.library = o.library;
        const RunResult v = r.run(ro);
        ro.library = false;
        save_outputs(v, out);
        const Verdict vv = compare_runs(a.outputs, v.outputs, o.tol, out, "A-vs-variant", true);
        const std::string av_stores = stored_difference(a, v);
        json::Value vj = verdict_json(vv, vv.match && v.writes.empty() && av_stores.empty(),
                                      "shader variant against the captured modules (reported, not part of the exit status)");
        if (!a.stored.empty()) {
            vj.set("written_buffers", static_cast<std::uint64_t>(a.stored.size()));
            vj.set("written_bytes", static_cast<std::uint64_t>(stored_bytes(a)));
            vj.set("stores", av_stores.empty() ? std::string("identical") : av_stores);
        }
        if (!a.stored.empty()) std::printf("  A vs variant stores: %s (%zu bytes in %zu written buffers)\n", av_stores.empty() ? "identical" : av_stores.c_str(),
                                           stored_bytes(a), a.stored.size());
        // The variant again with the sentinel sink: a variant that reads guest
        // memory the capture lacks (its own page-table walks) changes output.
        {
            ro.name = "variant-sentinel";
            ro.sentinel = true;
            const RunResult vs = r.run(ro);
            ro.sentinel = false;
            const Verdict svv = compare_runs(v.outputs, vs.outputs, exact, out, "variant-vs-variant-sentinel", false);
            json::Value vsj = verdict_json(svv, svv.identical, "the variant with the sink page filled: a change means it reads guest memory the capture lacks");
            comparisons.set("variant_vs_variant_sentinel", vsj);
            std::printf("  variant with the sentinel sink: %s\n", svv.identical ? "output unchanged" : "OUTPUT CHANGED (the variant reads memory the capture lacks)");
        }
        if (!o.ps.empty()) vj.set("ps", o.ps.string());
        if (!o.vs.empty()) vj.set("vs", o.vs.string());
        if (o.library) vj.set("library", true);
        comparisons.set("A_vs_variant", vj);
        runs.set("variant", writes_json(v));
    }

    if (o.bench) {
        // Alternating batches of the same draw from the same inputs: the only
        // difference between the two series is the pixel shader.
        json::Value bench = json::Value::make_object();
        bench.set("draws_per_batch", o.bench);
        bench.set("note", "one submission per batch, GPU timestamps around the draws only; targets accumulate across the batch "
                          "and are not compared; capture/replay builds are not the game's timing runs");
        // Warm-up batches are discarded (pipeline finalisation, caches), and the
        // order alternates so drift over the run favours neither pipeline.
        constexpr int kWarmup = 3, kBatches = 16;
        std::vector<double> translated, variant;
        for (int k = 0; k < kWarmup + kBatches; ++k) {
            RunOptions b;
            b.name = "bench";
            b.timed = true;
            b.repeat = o.bench;
            const auto time = [&](VkPipeline p) {
                b.pipeline = p;
                return r.run(b).gpu_ms / o.bench;
            };
            double tc = 0.0, tv = 0.0;
            if (variant_pipeline && k % 2) {
                tv = time(variant_pipeline);
                tc = time(captured);
            } else {
                tc = time(captured);
                if (variant_pipeline) tv = time(variant_pipeline);
            }
            if (k < kWarmup) continue;
            translated.push_back(tc);
            if (variant_pipeline) variant.push_back(tv);
        }
        bench.set("warmup_batches_discarded", kWarmup);
        const auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v.empty() ? 0.0 : v[v.size() / 2];
        };
        const auto series = [](const std::vector<double>& v) {
            json::Value a = json::Value::make_array();
            for (double x : v) a.push(x);
            return a;
        };
        bench.set("captured_ms_per_draw", series(translated));
        bench.set("captured_median_ms", median(translated));
        std::printf("  bench: captured modules %.4f ms per draw (median of %zu batches of %u)\n", median(translated), translated.size(), o.bench);
        if (!variant.empty()) {
            bench.set("variant_ms_per_draw", series(variant));
            bench.set("variant_median_ms", median(variant));
            std::printf("  bench: variant pixel shader %.4f ms per draw\n", median(variant));
        }
        report.set("bench", bench);
    }

    json::Value spirv = json::Value::make_object();
    bool spirv_ok = true;
    for (const auto& [stage, result] : r.spirv_checks) {
        spirv.set(stage, result);
        spirv_ok = spirv_ok && result.rfind("invalid", 0) != 0 && result.rfind("word ", 0) != 0;
    }
    report.set("spirv_validation", spirv);
    report.set("run_input_writes", runs);
    report.set("comparisons", comparisons);
    json::Value val = json::Value::make_array();
    for (const std::string& msg : dev.validation) val.push(msg);
    report.set("validation_messages", val);

    const bool no_writes = a.writes.empty() && b.writes.empty() && sentinel.writes.empty();
    const bool pass = deterministic && tracked && self_ok && live_ok && no_writes && spirv_ok && dev.validation.empty();
    json::Value verdict = json::Value::make_object();
    verdict.set("replay_deterministic", deterministic);
    verdict.set("matches_live", o.live ? json::Value(live_ok) : json::Value("not checked"));
    verdict.set("sentinel_sink_left_output_unchanged", tracked);
    verdict.set("no_input_writes", no_writes);
    verdict.set("self_tests_detect_perturbations", self_ok);
    verdict.set("spirv_valid", spirv_ok);
    verdict.set("no_validation_messages", dev.validation.empty());
    verdict.set("pass", pass);
    report.set("verdict", verdict);
    const std::string text = json::dump(report);
    write_file(out / "report.json", reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
    std::printf("  A vs B: %s (%llu texels different)\n", deterministic ? "match" : "DIFFER", static_cast<unsigned long long>(ab.texels_different));
    if (o.live) std::printf("  A vs live: %s\n", live_ok ? "match" : "DIFFER");
    std::printf("  sentinel sink: %s\n", tracked ? "output unchanged" : "OUTPUT CHANGED (untracked reads) OR INPUTS WRITTEN");
    std::printf("  self-tests: %s\n", self_ok ? "perturbations detected" : "FAILED");
    std::printf("%s: %s\n", (out / "report.json").string().c_str(), pass ? "pass" : "FAIL");
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const Options o = parse_args(argc, argv);
    try {
        return replay_main(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "drawreplay: %s\n", e.what());
        return 2;
    }
}
