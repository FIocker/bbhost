#include "engine/mouse_camera.h"

// What DS3 does, read from its executable (the 1.15 in-game dump; the Steam
// build keeps these functions encrypted on disk):
//
// - DLUID::MouseDevice polls DirectInput (GetDeviceState into DIMOUSESTATE2)
//   and converts lX/lY to float unscaled: the camera works in raw counts.
// - The game's pad object reads them as two logical axes, signs them by the
//   PC key config's mouse invert bytes (-1 when not inverted), and scales
//   them by ((sensitivity * 0.1) * 1.5 + 0.5) * 0.2, the sensitivity a float
//   0..10 defaulting to 5 (CSPcKeyConfig +0x710, +0x715, +0x716). They are 0
//   while the mouse is a pointer (a menu up).
// - PadManipulator keeps them beside the stick, not in it (+0xb0 against the
//   stick's +0x80).
// - ChrExFollowCam's update, after the stick has turned the camera:
//       if (stick moved this frame) skip;             // the stick has priority
//       pitch -= mouse.pitch * (pi / 180);
//       yaw   -= mouse.yaw   * (pi / 180);
//       clamp pitch to the stick path's limits; camera input present
//   and that is all - the angle is per count, whatever the frame time.
// - Locked on, a frame whose movement is over 50 counts switches the target
//   in that direction, then waits 0.5 s unless the movement drops under 25
//   counts first (the lock-on manager, sub_14085eac0 in the dump).
//
// Bloodborne's ChrExFollowCam::Update (sub_183ac60) is the same function
// without the mouse. Its free camera turns by the stick in the block at
// 0x183cd75 - angles += dt * stick * speed, both wrapped, pitch clamped to
// [+0x1f0, +0x1ec] - and stores them at 0x183ce54; the next block, 0x183ce67,
// restarts the camera's return to centre when the frame had input
// ([rbp-0x3c4]). DS3's step goes between the two. Lock-on and the scripted
// camera turns reach 0x183ce67 by other paths and never through the store, so
// the store is also how this knows the free camera ran.

#include "core/elf.h"
#include "core/memory.h"
#include "core/thunk.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "guest_abi.h"
#include "host/settings.h"
#include "host/window.h"
#include "log.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

// The stubs at the free camera's stores (below) reach the host through these.
extern "C" {
void* g_mouse_camera_host = nullptr;          // thunk_wrap(free_camera_hook)
std::uint64_t g_mouse_camera_resume = 0;      // the guest's 0x183ce67
std::uint64_t g_mouse_camera_updates = 0;     // camera updates so far
std::uint64_t g_mouse_camera_turned = 0;      // the last update whose free camera ran
void bb_mouse_camera_stub();
void bb_mouse_camera_ramp_stub();
}

// The two stores the hook displaced, then the host with every register the
// game's code could still need around it - the SysV ones a call may clobber,
// and all sixteen xmm (the thunk uses xmm8-15 as scratch; the camera keeps its
// stick vector in xmm10) - and the new pitch back in xmm0, which the next
// block stores as the return-to-centre reference. The update is not a leaf,
// so nothing of its lives below rsp to be pushed over.
asm(R"(
    .text
    .globl bb_mouse_camera_stub
bb_mouse_camera_stub:
    vmovss %xmm0, 0x140(%r13)
    vextractps $1, %xmm1, 0x144(%r13)
    push %rax
    push %rcx
    push %rdx
    push %rsi
    push %rdi
    push %r8
    push %r9
    push %r10
    push %r11
    push %rbx
    mov %rsp, %rbx
    and $-16, %rsp
    sub $256, %rsp
    vmovdqu %xmm0, 0(%rsp)
    vmovdqu %xmm1, 16(%rsp)
    vmovdqu %xmm2, 32(%rsp)
    vmovdqu %xmm3, 48(%rsp)
    vmovdqu %xmm4, 64(%rsp)
    vmovdqu %xmm5, 80(%rsp)
    vmovdqu %xmm6, 96(%rsp)
    vmovdqu %xmm7, 112(%rsp)
    vmovdqu %xmm8, 128(%rsp)
    vmovdqu %xmm9, 144(%rsp)
    vmovdqu %xmm10, 160(%rsp)
    vmovdqu %xmm11, 176(%rsp)
    vmovdqu %xmm12, 192(%rsp)
    vmovdqu %xmm13, 208(%rsp)
    vmovdqu %xmm14, 224(%rsp)
    vmovdqu %xmm15, 240(%rsp)
    mov %r13, %rdi
    lea -0x3c4(%rbp), %rsi
    vmovaps %xmm10, %xmm0
    vmovshdup %xmm10, %xmm1
    mov g_mouse_camera_host(%rip), %rax
    call *%rax
    vmovdqu 0(%rsp), %xmm0
    vmovdqu 16(%rsp), %xmm1
    vmovdqu 32(%rsp), %xmm2
    vmovdqu 48(%rsp), %xmm3
    vmovdqu 64(%rsp), %xmm4
    vmovdqu 80(%rsp), %xmm5
    vmovdqu 96(%rsp), %xmm6
    vmovdqu 112(%rsp), %xmm7
    vmovdqu 128(%rsp), %xmm8
    vmovdqu 144(%rsp), %xmm9
    vmovdqu 160(%rsp), %xmm10
    vmovdqu 176(%rsp), %xmm11
    vmovdqu 192(%rsp), %xmm12
    vmovdqu 208(%rsp), %xmm13
    vmovdqu 224(%rsp), %xmm14
    vmovdqu 240(%rsp), %xmm15
    mov %rbx, %rsp
    pop %rbx
    pop %r11
    pop %r10
    pop %r9
    pop %r8
    pop %rdi
    pop %rsi
    pop %rdx
    pop %rcx
    pop %rax
    vmovss 0x140(%r13), %xmm0
    jmp *g_mouse_camera_resume(%rip)

# The stick's fast turn - a stick held over, ramping up to the high speed -
# is the free camera too, with its own two stores of pitch and a jump to
# 0x183ce67. The stick has the camera there, so the mouse waits, as in DS3;
# the stub only notes that the free camera ran. rax is dead: the next block
# loads eax first.
    .globl bb_mouse_camera_ramp_stub
bb_mouse_camera_ramp_stub:
    vmovss %xmm0, 0x140(%r13)
    mov g_mouse_camera_updates(%rip), %rax
    mov %rax, g_mouse_camera_turned(%rip)
    jmp *g_mouse_camera_resume(%rip)
)");

namespace {

using ull = unsigned long long;
using Clock = std::chrono::steady_clock;

// NS_SPRJ::ChrExFollowCam::Update, the player's camera in play: push rbp;
// mov rbp, rsp; push r15..rbx; sub rsp, 0x3d8 (20 bytes).
constexpr std::uint64_t kUpdate = 0x183ac60;
constexpr std::uint8_t kUpdatePrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                            0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0xd8, 0x03, 0x00, 0x00};
// Its free camera's store: vmovss [r13+0x140], xmm0; vextractps [r13+0x144],
// xmm1, 1 (19 bytes), reached only from the stick step before it.
constexpr std::uint64_t kTurnStore = 0x183ce54;
constexpr std::uint8_t kTurnStoreBytes[] = {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x40, 0x01, 0x00, 0x00, 0xc4,
                                            0xc3, 0x79, 0x17, 0x8d, 0x44, 0x01, 0x00, 0x00, 0x01};
constexpr std::uint64_t kTurnResume = 0x183ce67;
// The fast turn's two exits: vmovss [r13+0x140], xmm0; jmp 0x183ce67.
constexpr std::uint64_t kRampExits[] = {0x183f9bc, 0x183fabf};
constexpr std::uint8_t kRampExitBytes[2][14] = {
    {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x40, 0x01, 0x00, 0x00, 0xe9, 0x9d, 0xd4, 0xff, 0xff},
    {0xc4, 0xc1, 0x7a, 0x11, 0x85, 0x40, 0x01, 0x00, 0x00, 0xe9, 0x9a, 0xd3, 0xff, 0xff}};
constexpr std::size_t kPitch = 0x140, kYaw = 0x144, kPitchMax = 0x1ec, kPitchMin = 0x1f0;

constexpr float kRadiansPerDegree = 0.0174532924f;  // DS3's constant, the float of pi/180
// The lock-on switch, in counts per frame at the 60 fps DS3 runs at; here as
// rates, so Bloodborne's 30 fps asks for the same movement.
constexpr float kFlickSwitch = 50.0f * 60.0f, kFlickRearm = 25.0f * 60.0f;
constexpr float kFlickCooldown = 0.5f;  // seconds, as DS3 sets it after a switch
// How long the stick shows a flick: two frames at 30 fps, enough for the game
// to see a push and a release.
constexpr auto kFlickHold = std::chrono::milliseconds(80);

bool g_installed = false;
const bool g_log = [] {
    const char* e = std::getenv("BBHOST_MOUSE_CAMERA_LOG");
    return e && e[0] == '1';
}();

std::mutex g_mu;  // the camera's frame and the flick, from the camera's thread and the pad's
float g_dx = 0.0f, g_dy = 0.0f;       // this update's mouse counts, until the free camera takes them
Clock::time_point g_last;
float g_cooldown = 0.0f;
float g_flick_x = 0.0f, g_flick_y = 0.0f;
Clock::time_point g_flick_until;

std::atomic<std::uint64_t> g_turns{0}, g_stick_frames{0}, g_flicks{0};
std::atomic<double> g_counts{0.0};

// Every camera update, first: this frame's mouse movement, and - when the
// free camera did not run in the frame before, which is lock-on - DS3's
// target switch.
GUEST_ABI std::int64_t update_hook(std::uint64_t, const std::uint64_t*) {
    float dx = 0.0f, dy = 0.0f;
    host_mouse_take_camera(dx, dy);
    const Clock::time_point now = Clock::now();
    std::lock_guard<std::mutex> lock(g_mu);
    float dt = __atomic_load_n(&g_mouse_camera_updates, __ATOMIC_RELAXED) ? std::chrono::duration<float>(now - g_last).count() : 0.0f;
    if (dt > 0.25f) dt = 0.25f;  // a load or a pause is not a fast mouse
    g_last = now;
    const std::uint64_t updates = __atomic_load_n(&g_mouse_camera_updates, __ATOMIC_RELAXED);
    const bool turned_before = updates && __atomic_load_n(&g_mouse_camera_turned, __ATOMIC_RELAXED) == updates;
    // When the free camera stops (lock-on, a scripted turn) and comes back.
    static bool was_turning = false;
    static int switches = 0;
    if (updates > 1 && turned_before != was_turning && (g_log || switches < 8)) {
        ++switches;
        host_log("mouse camera: the free camera %s", turned_before ? "is back" : "stopped (locked on, or the game turns it)");
    }
    was_turning = turned_before;
    __atomic_store_n(&g_mouse_camera_updates, updates + 1, __ATOMIC_RELAXED);
    g_dx = dx;
    g_dy = dy;
    if (turned_before || dt <= 0.0f) return 0;
    const float m = std::sqrt(dx * dx + dy * dy);
    const float rate = m / dt;
    if (rate < kFlickRearm) {
        g_cooldown = 0.0f;
    } else {
        g_cooldown = g_cooldown - dt < 0.0f ? 0.0f : g_cooldown - dt;
    }
    if (g_cooldown <= 0.0f && rate > kFlickSwitch) {
        g_flick_x = dx / m;
        g_flick_y = dy / m;
        g_flick_until = now + kFlickHold;
        g_cooldown = kFlickCooldown;
        const std::uint64_t n = g_flicks.fetch_add(1, std::memory_order_relaxed);
        if (g_log || n < 4) {
            host_log("mouse camera: a flick of %.0f counts (%.0f a second) switches the lock-on target, toward %+.2f,%+.2f",
                     m, rate, g_flick_x, g_flick_y);
        }
    }
    return 0;
}

// The free camera, after the stick has turned it: DS3's mouse step.
GUEST_ABI void free_camera_hook(std::uint8_t* cam, std::uint8_t* input, float stick_pitch, float stick_yaw) {
    float dx = 0.0f, dy = 0.0f;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        __atomic_store_n(&g_mouse_camera_turned, __atomic_load_n(&g_mouse_camera_updates, __ATOMIC_RELAXED), __ATOMIC_RELAXED);
        dx = g_dx;
        dy = g_dy;
        g_dx = g_dy = 0.0f;
    }
    if (stick_pitch != 0.0f || stick_yaw != 0.0f) {
        if (dx != 0.0f || dy != 0.0f) g_stick_frames.fetch_add(1, std::memory_order_relaxed);
        static int logs = 0;
        if (g_log && logs < 60) {
            ++logs;
            float pitch = 0.0f, yaw = 0.0f;
            std::memcpy(&pitch, cam + kPitch, 4);
            std::memcpy(&yaw, cam + kYaw, 4);
            host_log("mouse camera: the stick (%+.2f,%+.2f) has the camera: pitch %.4f, yaw %.4f", stick_pitch, stick_yaw, pitch, yaw);
        }
        return;
    }
    if (dx == 0.0f && dy == 0.0f) return;
    const HostSettings hs = host_settings();
    const float per_count = mouse_camera_degrees_per_count(hs.mouse_sens);
    // DS3's signs: a count is negated unless the axis is inverted, and the
    // camera subtracts.
    const float mouse_pitch = (hs.mouse_invert_y ? 1.0f : -1.0f) * dy * per_count;
    const float mouse_yaw = (hs.mouse_invert_x ? 1.0f : -1.0f) * dx * per_count;
    float pitch = 0.0f, yaw = 0.0f, lo = 0.0f, hi = 0.0f;
    std::memcpy(&pitch, cam + kPitch, 4);
    std::memcpy(&yaw, cam + kYaw, 4);
    std::memcpy(&lo, cam + kPitchMin, 4);
    std::memcpy(&hi, cam + kPitchMax, 4);
    const float was_pitch = pitch, was_yaw = yaw;
    pitch -= mouse_pitch * kRadiansPerDegree;
    yaw -= mouse_yaw * kRadiansPerDegree;
    // The stick step's clamp: the lower limit if under it, else at most the upper.
    pitch = lo > pitch ? lo : (hi < pitch ? hi : pitch);
    std::memcpy(cam + kPitch, &pitch, 4);
    std::memcpy(cam + kYaw, &yaw, 4);
    *input = 1;
    const std::uint64_t n = g_turns.fetch_add(1, std::memory_order_relaxed);
    g_counts.store(g_counts.load(std::memory_order_relaxed) + std::sqrt(dx * dx + dy * dy), std::memory_order_relaxed);
    if (n == 0 || (g_log && n < 400)) {
        host_log("mouse camera: %+.0f,%+.0f counts at sensitivity %d (%.3f degrees a count): pitch %.4f -> %.4f, yaw %.4f -> %.4f",
                 dx, dy, hs.mouse_sens, per_count, was_pitch, pitch, was_yaw, yaw);
    }
}

}  // namespace

float mouse_camera_degrees_per_count(int sensitivity) {
    const float s = static_cast<float>(sensitivity < 0 ? 0 : sensitivity > 10 ? 10 : sensitivity);
    return ((s * 0.1f) * 1.5f + 0.5f) * 0.2f;
}

bool mouse_camera_installed() { return g_installed; }

void mouse_camera_install(ElfImage* image) {
    if (!image || image->sha256 != kEboot109Sha256) return;
    const auto at = [&](std::uint64_t bn) { return image->mem.slide + (bn - kPreferredGuestSlide); };
    auto* store = static_cast<std::uint8_t*>(guest_ptr(image->mem, at(kTurnStore)));
    if (!store || std::memcmp(store, kTurnStoreBytes, sizeof(kTurnStoreBytes)) != 0) {
        host_log("mouse camera: refused, 0x%llx is not the follow camera's store; the mouse turns it through the stick",
                 static_cast<ull>(kTurnStore));
        return;
    }
    if (!engine_prologue_hook(image, at(kUpdate), kUpdatePrologue, sizeof(kUpdatePrologue),
                              reinterpret_cast<void*>(&update_hook))) {
        host_log("mouse camera: refused, 0x%llx is not the follow camera's update; the mouse turns it through the stick",
                 static_cast<ull>(kUpdate));
        return;
    }
    g_mouse_camera_host = thunk_wrap(reinterpret_cast<void*>(&free_camera_hook));
    g_mouse_camera_resume = at(kTurnResume);
    const std::uint64_t lo = at(kTurnStore) & ~0xfffull, hi = (at(kTurnStore) + sizeof(kTurnStoreBytes) + 0xfff) & ~0xfffull;
    if (!guest_protect_rwx(&image->mem, lo, hi - lo)) {
        host_log("mouse camera: cannot write the follow camera's code; the mouse turns it through the stick");
        return;
    }
    const std::uint64_t dest = reinterpret_cast<std::uint64_t>(&bb_mouse_camera_stub);
    store[0] = 0xff;
    store[1] = 0x25;  // jmp [rip+0]
    std::memset(store + 2, 0, 4);
    std::memcpy(store + 6, &dest, 8);
    std::memset(store + 14, 0xcc, sizeof(kTurnStoreBytes) - 14);
    guest_protect_rx(&image->mem, lo, hi - lo);
    // The fast turn's exits, so that "the free camera did not run" is lock-on
    // or the game turning the camera, never the stick held over.
    const std::uint64_t ramp = reinterpret_cast<std::uint64_t>(&bb_mouse_camera_ramp_stub);
    for (int i = 0; i < 2; ++i) {
        auto* exit = static_cast<std::uint8_t*>(guest_ptr(image->mem, at(kRampExits[i])));
        const std::uint64_t elo = at(kRampExits[i]) & ~0xfffull, ehi = (at(kRampExits[i]) + 14 + 0xfff) & ~0xfffull;
        if (!exit || std::memcmp(exit, kRampExitBytes[i], 14) != 0 || !guest_protect_rwx(&image->mem, elo, ehi - elo)) {
            host_log("mouse camera: 0x%llx is not the fast turn's exit; a lock-on flick may also show while the stick is held "
                     "over", static_cast<ull>(kRampExits[i]));
            continue;
        }
        exit[0] = 0xff;
        exit[1] = 0x25;
        std::memset(exit + 2, 0, 4);
        std::memcpy(exit + 6, &ramp, 8);
        guest_protect_rx(&image->mem, elo, ehi - elo);
    }
    g_installed = true;
    host_log("mouse camera: as Dark Souls III turns it - %.3f degrees a count at sensitivity 5, after the stick and only "
             "without it; a flick switches the lock-on target",
             mouse_camera_degrees_per_count(5));
}

bool mouse_camera_flick(float& x, float& y) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (Clock::now() >= g_flick_until) return false;
    x = g_flick_x;
    y = g_flick_y;
    return true;
}

void mouse_camera_report() {
    if (!g_installed) return;
    host_log("mouse camera: %llu frames turned by the mouse (%.0f counts), %llu where the stick had the camera, %llu lock-on "
             "flicks; %llu camera updates",
             static_cast<ull>(g_turns.load()), g_counts.load(), static_cast<ull>(g_stick_frames.load()),
             static_cast<ull>(g_flicks.load()), static_cast<ull>(__atomic_load_n(&g_mouse_camera_updates, __ATOMIC_RELAXED)));
}
