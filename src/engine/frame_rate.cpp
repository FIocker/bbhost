#include "guest_abi.h"
#include "engine/frame_rate.h"
#include "engine/np_test.h"
#include "engine/event_flags.h"
#include "engine/player_data.h"
#include "engine/lua_events.h"
#include "engine/world_chr.h"
#include "engine/params.h"
#include "engine/sprj_flipper.h"
#include "engine/loading.h"
#include "engine/change_appearance.h"
#include "engine/rebirth.h"

#include "core/elf.h"
#include "core/memory.h"
#include "core/portable.h"
#include "engine/addr.h"
#include "engine/graphics_patch.h"
#include "hle/hle.h"
#include "hle/modules.h"
#include "host/frame_stats.h"
#include "host/plugins.h"
#include "host/options.h"
#include "host/settings.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

std::atomic<float> g_time_scale{1.0f};

constexpr std::uint64_t kFrameManager = 0x2434770;
constexpr std::uint8_t kFrameManagerPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                                                  0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x38};
constexpr std::uint32_t kModeSixty = 2;

// The game's fixed steps, converted for 60: every site the community's "60
// FPS++" patch list for 1.09 (shadPS4's ps4_cheats) gives a float constant,
// as the value it had and the value it gets. 86 are 1/30 s steps made 1/60,
// 29 are 30 steps a second made 60; the rest are per-frame weights and
// decays retuned to the shorter frame (0.1 -> 0.05, 0.95 -> 0.983...). Some
// are immediates in instructions, some constants in .rodata; each is checked
// for its old value before anything is written, and one mismatch leaves the
// game at 30. The list's code rewrites are kSixtySites below.
//
// Not every 1/30 in the game is a step. A duration the data gives in frames
// (an event's "wait N frames", a light's fade frames, param fields) becomes
// seconds as `vcvtsi2ss; vmulss [1/30]`, and what holds it counts down by
// the frame's real seconds (the event wait: sub_1bc8630 makes it, sub_1bb5830
// subtracts the step) - so 1/30 is right at any frame rate. An earlier
// version of the list pointed 22 of those loads at the measured frame time
// and set 15 of their constants to 1/60 (halving the durations at 60), and
// two more conversions (0x4caf1c0, 0x4d27284) to 1/15 (doubling them): one
// "wait N frames" command ran four times as long by one path as by the
// other. Kyo's later list leaves them all at 1/30, and so does this table.
// Three per-frame amounts it also leaves at 1/30 are made per-frame for 60
// here instead: the debug camera's step (0x1972a80), the target bank's
// refresh accumulator (0x4d2577c), and a camera smoothing weight
// (0x4d25e94: 1 - sqrt(1 - 1/30), the same decay over two frames). Two the
// later list puts back to 1/30 stay 1/60 as this build has always run them:
// the physics world's step (0x0f383ca) and 0x4d29170's per-call step.
struct Constant {
    std::uint32_t at;
    std::uint32_t was, now;
};
constexpr Constant kSixtyConstants[] = {
    {0x0a3fc3f, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x0f383ca, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x12b5458, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x12b55b6, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x13734f3, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x13ef51c, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x13ef6ec, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x16efe5c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x183a4c3, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a4cd, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a4d7, 0x3e99999a, 0x3d4ccccd},  // 0.3 -> 0.05
    {0x183a4e1, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a4f5, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a4ff, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a509, 0x3cf5c28f, 0x3c75c28f},  // 0.03 -> 0.015
    {0x183a527, 0x3dcccccd, 0x3e4ccccd},  // 0.1 -> 0.2
    {0x183a531, 0x3e99999a, 0x3e4ccccd},  // 0.3 -> 0.2
    {0x183a53b, 0x3e99999a, 0x3e19999a},  // 0.3 -> 0.15
    {0x183a649, 0x3f860a92, 0x3f32b8c2},  // 1.0472 -> 0.698132
    {0x183a653, 0x3d75c28f, 0x3d23d70a},  // 0.06 -> 0.04
    {0x183a65d, 0x3dcccccd, 0x3e19999a},  // 0.1 -> 0.15
    {0x183a8c1, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a8cb, 0x3e800000, 0x3d4ccccd},  // 0.25 -> 0.05
    {0x183a8d5, 0x3e99999a, 0x3d4ccccd},  // 0.3 -> 0.05
    {0x183a8df, 0x3e99999a, 0x3d4ccccd},  // 0.3 -> 0.05
    {0x183a8f3, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x183a8fd, 0x3e800000, 0x3d4ccccd},  // 0.25 -> 0.05
    {0x183a907, 0x3cf5c28f, 0x3c75c28f},  // 0.03 -> 0.015
    {0x183a925, 0x3ecccccd, 0x3e4ccccd},  // 0.4 -> 0.2
    {0x183a92f, 0x3ecccccd, 0x3e4ccccd},  // 0.4 -> 0.2
    {0x183a939, 0x3e99999a, 0x3e19999a},  // 0.3 -> 0.15
    {0x183aa47, 0x3f860a92, 0x3f32b8c2},  // 1.0472 -> 0.698132
    {0x183aa51, 0x3d75c28f, 0x3d23d70a},  // 0.06 -> 0.04
    {0x183aa5b, 0x3e99999a, 0x3e19999a},  // 0.3 -> 0.15
    {0x187305d, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x187458d, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x187afec, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x187b0ad, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x1880cfd, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x1880e1a, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x18b254d, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x18b382d, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x1972a80, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ab67ef, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x1bf9b74, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1c0ca9a, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1c9f889, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x1ca975a, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x1ca9892, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x1daa65e, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea4003, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea40a0, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea4146, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea49ce, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea4bda, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea5248, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea5476, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea552d, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea57c0, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea5ac5, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea6015, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea6045, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea6275, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea62a5, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ea632f, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1eb540b, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1eb54dc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1eb90aa, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1eb9117, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ebb111, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ebf0d3, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec3f69, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec3f9a, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec3fd8, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec4009, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec404e, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec40a4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec40e0, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec4136, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec4167, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec41a5, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1ec41dc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1f032f2, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1f03762, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1f6aa32, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x1f96d9a, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x200d8e2, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x200e142, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x200e2d4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x200fa52, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x20119d2, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2012652, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x20127cc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2012932, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2012ac4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2012c32, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x203acb7, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x21bb004, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x222bcd7, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x23dd026, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x23dede6, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x2418e39, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2418f38, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x243460c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x243485f, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x2434883, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x24348b8, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x26fb39f, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4befdcc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4c88688, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4c9f358, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4cad65c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4cad660, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4cad664, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4cad7fc, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4cadc08, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4cadd28, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4ccc3f4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d02f10, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4d02f14, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4d02f18, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4d02f1c, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4d02f70, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4d02f74, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4d02f78, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4d02f7c, 0x41f00000, 0x42700000},  // 30 -> 60
    {0x4d0b5b0, 0x3f733333, 0x3f7ba901},  // 0.95 -> 0.983048
    {0x4d0b5b4, 0x3f733333, 0x3f7ba901},  // 0.95 -> 0.983048
    {0x4d0b5b8, 0x3f733333, 0x3f7ba901},  // 0.95 -> 0.983048
    {0x4d0b5bc, 0x3f733333, 0x3f7ba901},  // 0.95 -> 0.983048
    {0x4d2577c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d25e68, 0x3dcccccd, 0x3ccccccd},  // 0.1 -> 0.025
    {0x4d25e94, 0x3d088889, 0x3c89b0c3},  // 0.0333333 -> 0.0168070
    {0x4d265fc, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d26618, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d26c9c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d270e4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d27554, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d275f4, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d27720, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d27c4c, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d27fd4, 0xbdcccccd, 0xbd4ccccd},  // -0.1 -> -0.05
    {0x4d27fe0, 0x3dcccccd, 0x3d4ccccd},  // 0.1 -> 0.05
    {0x4d28418, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d28e98, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d29170, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d29b80, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
    {0x4d29e74, 0x3d088889, 0x3c888889},  // 0.0333333 -> 0.0166667
};

// The same list's other value changes: a frame count doubled (10 -> 20 at
// 0x192c3ff's movl, 5 -> 10 at 0x1929675's mov edx), one load moved from a
// 1/30 constant that stays 1/30 for its other readers to one this table
// makes 1/60 (0x1915fab), a worker's usleep(33000) halved (0x25629f0), and
// the frame-time manager's catch-up allowance zeroed (its movabs rcx,
// 0x1e0000000n into +0x268/+0x26c) - the last two inside the function ours
// replaces, where they only matter with BBHOST_FRAME_SOURCE=0.
struct Bytes {
    std::uint32_t at;
    std::uint8_t n;
    std::uint8_t was[10], now[10];
};
constexpr Bytes kSixtyBytes[] = {
    {0x192c405, 4, {0x0a, 0, 0, 0}, {0x14, 0, 0, 0}},
    {0x1929676, 4, {0x05, 0, 0, 0}, {0x0a, 0, 0, 0}},
    {0x1915faf, 4, {0xb1, 0x03, 0x41, 0x03}, {0xbd, 0x31, 0x41, 0x03}},
    {0x25629f1, 4, {0xe8, 0x80, 0, 0}, {0x74, 0x40, 0, 0}},
    {0x2434863, 10, {0x48, 0xb9, 0x01, 0, 0, 0, 0x1e, 0, 0, 0}, {0x48, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0}},
    {0x2434887, 10, {0x48, 0xb9, 0, 0, 0, 0, 0x1e, 0, 0, 0}, {0x48, 0xb9, 0, 0, 0, 0, 0, 0, 0, 0}},
    // The sound system's thread (MagicOrchestra over FMOD, its loop 0x2c00e30)
    // updates every MainThreadUpdateTime us, read from the text config at
    // 0x4b33b40 into +0x8d4 (sub_2bf67f0; the engine's own default is 16666).
    // Bloodborne ships 33333: at 60 the game posts two frames of sound
    // requests per update, and some attack sounds played half as often as at
    // 30 (2026-09-30, tmp/sound60). The digits, in place.
    {0x4b33c51, 5, {'3', '3', '3', '3', '3'}, {'1', '6', '6', '6', '6'}},
};

// The rest of that list (Kyo's "60 FPS++" for 1.09), generated with its old
// bytes from the eboot by tools/sixty_fps_sites.py; the constants above are
// left out of it. What it does:
//  - the measured frame time, clamped to 1/60..1/30 by a cave over a panic
//    path (0x2418e3d, the flipper's +0x264 against +0x18), stored in a free
//    slot (0x575a908) that the loads of shared 1/30 steps the list names read, and
//    multiplied in where 0x21bc181 had a fixed step - those systems step by
//    the frame that ran rather than by 1/30 (the list's author credits the
//    cloth and jumping fixes at 60 to these);
//  - two runs of parameter reads (0x183ff29, 0x1840799, beside the camera's
//    field-of-view clamp) replaced by stores of the list's 60 fps values;
//  - a few constants and table bytes (0x4cf6422, 0x47df586, 0x3abc2xx), a
//    count forced to 1 (0xb6e67a), the motion blur's sample check (0xfbc40f).
// Left out, each with its reason in the generator: the frame-time manager's
// own edits (ours replaces that function), a frame-mode store, the mode-name
// pointer table, the timed wait turned into a return, and the sound thread's
// sleep cap (a cave over a live function) - that thread's update time is in
// kSixtyBytes instead. BBHOST_SIXTY_SITES=0 leaves them all out.
struct Site {
    std::uint32_t at;
    const char* was;
    const char* now;
};
constexpr Site kSixtySites[] = {
#include "engine/sixty_fps_sites.inc"
};

std::vector<std::uint8_t> hex_bytes(const char* h) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; h[i] && h[i + 1]; i += 2) {
        auto nib = [](char c) { return static_cast<std::uint8_t>(c <= '9' ? c - '0' : (c | 0x20) - 'a' + 10); };
        out.push_back(static_cast<std::uint8_t>(nib(h[i]) << 4 | nib(h[i + 1])));
    }
    return out;
}

int g_game_fps = 30;
std::size_t g_sites_applied = 0;
std::atomic<std::uint64_t> g_frames{0};

template <typename T>
void put(std::uint64_t a, T v) {
    std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(a)), &v, sizeof(v));
}
template <typename T>
T get(std::uint64_t a) {
    T v;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(a)), sizeof(v));
    return v;
}

// --- SprjFlipper::Update as source -----------------------------------------
//
// What the eboot's 0x2434770 does, in order, read from its decompile
// (Binary Ninja) with sprj_flipper.h's field names: pick the mode (the
// secondary one for one frame when requested), write the frame-advance
// policy and the target interval for it, take the one-shot overrides, sample
// the clock, shorten this frame's interval when the last frames ran late
// (down to a third), sleep to within 5 ms of the deadline and spin the rest,
// record the frame in the 32-entry history and the 16-sample FPS average,
// decide whether the next frame is "behind", and ask the window for 60 Hz
// once when told to. sub_13e3980, which chose the foreground history count
// and the sleeping wait, returns 1 in this build, so only that path exists.
//
// Ours differs in one thing: which mode runs is the host's choice (the
// Frame cap option), not the game's fields, and at 60 the game's own
// one-shot resets to 30 (the multiplayer route, the "reset to 30" request)
// do not apply. The clock and the sleep are the host's; the arithmetic is
// the eboot's, float for float, so compare mode can demand equality.

struct FlipClock {
    // The clock as the eboot reads it: gettimeofday in microseconds. In a
    // replay (compare mode) it returns the release time the guest reached,
    // and the sleep does nothing.
    virtual std::int64_t now() = 0;
    virtual void sleep_us(std::int64_t us) = 0;
    // How close to the deadline the wait sleeps before it spins: the eboot's
    // 5 ms, or less for a host sleep that keeps time better (RealClock).
    virtual float spin_margin_s() { return 0.00499999989f; }
    // Between two reads of the clock while spinning.
    virtual void spin_pause() {}
    virtual ~FlipClock() = default;
};

struct RealClock final : FlipClock {
    std::int64_t now() override {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
    void sleep_us(std::int64_t us) override { host_sleep_us(static_cast<std::uint64_t>(us)); }
    // The eboot sleeps to 5 ms before the deadline and spins the rest on the
    // clock: a fifth of the main thread's CPU at 60 fps went there (the
    // 2026-10-04 profile), taken from its SMT sibling. Sleeping to 2 ms keeps
    // most of that back; 1 ms cost flip-interval p95 a millisecond on a busy
    // machine, the sleep overshooting. The spin pauses between reads.
    float spin_margin_s() override { return 0.002f; }
    void spin_pause() override {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#endif
    }
};

struct ReplayClock final : FlipClock {
    std::int64_t at;
    explicit ReplayClock(std::int64_t t) : at(t) {}
    std::int64_t now() override { return at; }
    void sleep_us(std::int64_t) override {}
};

// The unsigned-to-float conversion the compiler emitted for the frame's
// microseconds (a negative count halves and doubles); the count is never
// negative in practice, so this is float(int64).
inline float us_to_float(std::int64_t us) {
    if (us >= 0) return static_cast<float>(us);
    const auto u = static_cast<std::uint64_t>(us);
    const float h = static_cast<float>((u >> 1) | (u & 1));
    return h + h;
}

inline std::int64_t seconds_to_us(float s) {
    const double d = static_cast<double>(s) * 1000000.0;  // the eboot converts to double first
    return d >= 9223372036854775808.0 ? static_cast<std::int64_t>(d - 9223372036854775808.0) ^ INT64_MIN
                                      : static_cast<std::int64_t>(d);
}

// The window object at data_5940e00: told once to run at 60 Hz when the
// flipper is asked to (+0x4c, +0x4d flags and +0x50 = 60).
constexpr std::uint64_t kWindowObjectSlot = 0x5940e00;
std::uint64_t g_slide = 0;
// This frame runs with no target (a loading screen, engine/loading.h); set by
// the hook for its own update only, never for compare mode's replay.
bool g_fast_target = false;
// BBHOST_BENCH_UNCAPPED=1 (with BBHOST_UNCAP=1, which completes flips at
// once): every frame runs with no target - a measure of how fast the machine
// can run the game's frames, not a way to play it: the steps the 60 FPS++
// list fixed at 1/60 then go by faster than real time. Stand still to
// measure.
const bool g_bench_uncapped = [] {
    const char* e = std::getenv("BBHOST_BENCH_UNCAPPED");
    return e && e[0] == '1';
}();

void sprj_flipper_update(SprjFlipper* f, FlipClock& clock, bool sixty) {
    f->secondary_mode_active = f->secondary_mode_requested;
    f->secondary_mode_requested = 0;
    std::uint32_t mode = f->secondary_mode_active ? f->secondary_flip_mode_raw : f->primary_flip_mode_raw;
    if (sixty) mode = static_cast<std::uint32_t>(SprjFlipMode::Fps60WithoutSkip);
    if (mode > 4) mode = 0;
    constexpr float kThirtieth = 0.0333333351f, kSixtieth = 0.0166666675f;  // 0x3d088889, 0x3c888889
    switch (mode) {
        case 0: f->frame_advance_count = 2; f->allow_frame_skip = 1; f->target_frame_seconds = kThirtieth; f->foreground_history_count = 1; f->background_history_count = 30; break;
        case 1: f->frame_advance_count = 2; f->allow_frame_skip = 1; f->target_frame_seconds = kThirtieth; f->foreground_history_count = 0; f->background_history_count = 30; break;
        case 2: f->frame_advance_count = 1; f->allow_frame_skip = 1; f->target_frame_seconds = kSixtieth; f->foreground_history_count = 0; f->background_history_count = 30; break;
        case 3: f->frame_advance_count = 1; f->allow_frame_skip = 0; f->target_frame_seconds = kThirtieth; f->foreground_history_count = 1; f->background_history_count = 30; break;
        default: f->frame_advance_count = 1; f->allow_frame_skip = 0; f->target_frame_seconds = kThirtieth; f->foreground_history_count = 0; f->background_history_count = 30; break;
    }
    if (sixty) {
        // The 60 FPS patch list zeroes both history counts (its two movabs
        // edits); the same here.
        f->foreground_history_count = 0;
        f->background_history_count = 0;
        // A loading screen (engine/loading.h): no target, so no wait; the
        // 60 FPS++ frame-time cave (0x2418e3d) clamps the measured frame time
        // to [target, 1/30], so game time stays real time.
        if (g_fast_target) f->target_frame_seconds = 0.0f;
    }
    bool no_wait = false;
    if (!sixty && f->reset_to_30_fps_pending) {
        f->frame_advance_count = 1;
        f->allow_frame_skip = 0;
        f->target_frame_seconds = kThirtieth;
        f->foreground_history_count = 0;
        f->background_history_count = 0;
        f->reset_to_30_fps_pending = 0;
        no_wait = true;
    }
    if (!sixty) {
        if (f->override_foreground_history_count >= 0) f->foreground_history_count = static_cast<std::uint32_t>(f->override_foreground_history_count);
        if (f->override_background_history_count >= 0) f->background_history_count = static_cast<std::uint32_t>(f->override_background_history_count);
        if (f->reset_frame_history) {
            f->foreground_history_count = 0;
            f->background_history_count = 0;
            f->reset_frame_history = 0;
        }
        if (f->force_frame_advance) {  // one 64-bit store over both counts; the flag is not cleared here
            f->foreground_history_count = 0;
            f->background_history_count = 0;
        }
    } else {
        f->override_foreground_history_count = -1;
        f->override_background_history_count = -1;
        f->reset_frame_history = 0;
        f->force_frame_advance = 0;
        f->reset_to_30_fps_pending = 0;
    }
    f->previous_frame_microseconds = f->current_frame_microseconds;
    std::int64_t now = clock.now();
    f->current_frame_microseconds = static_cast<std::uint64_t>(now);
    // A plugin's time scale (frame_rate_set_time_scale): the game advances a
    // fixed step a frame, so a longer frame is slower time and a shorter one
    // faster, as far as the machine keeps up.
    std::int64_t interval_us = seconds_to_us(f->target_frame_seconds / g_time_scale.load(std::memory_order_relaxed));

    // The last frames that ran behind, up to the foreground count of them:
    // how much time they took together.
    const std::uint32_t allowance = f->foreground_history_count;
    std::uint32_t walked = 0;
    std::uint64_t behind_us = 0;
    for (int back = 0; walked < allowance; --back) {
        const SprjFlipperFrameHistoryEntry& e = f->frame_history[(f->frame_history_index + static_cast<std::uint32_t>(back)) & 0x1f];
        behind_us += e.elapsed_microseconds;
        ++walked;
        if (e.delayed_or_behind == 0) break;
    }
    if (f->frame_behind) {
        const std::uint64_t budget = static_cast<std::uint64_t>(walked + 1) * static_cast<std::uint64_t>(interval_us);
        if (behind_us + static_cast<std::uint64_t>(interval_us) > budget) {
            const std::uint64_t third = static_cast<std::uint64_t>(interval_us) / 3;
            if (budget <= behind_us) {
                interval_us = static_cast<std::int64_t>(third);
            } else {
                std::uint64_t rem = budget - behind_us;
                if (static_cast<std::uint64_t>(interval_us) < rem) rem = static_cast<std::uint64_t>(interval_us);
                if (third > rem) rem = third;
                interval_us = static_cast<std::int64_t>(rem);
            }
        }
    }
    if (!(no_wait || f->force_no_sleep)) {
        const float target_s = us_to_float(interval_us) / 1000000.0f;
        const std::int64_t prev = static_cast<std::int64_t>(f->previous_frame_microseconds);
        float remaining = target_s - us_to_float(now - prev) / 1000000.0f;
        const float margin = clock.spin_margin_s();
        while (!(0.0f > remaining)) {
            const float left = std::min(target_s, remaining);
            if (left <= 0.0f) break;
            if (!(left <= 0.00100000005f) && !(left <= margin)) {
                const auto us = static_cast<std::int32_t>((left - margin) * 1000.0f * 1000.0f);
                if (us > 0) clock.sleep_us(us);
            } else {
                clock.spin_pause();
            }
            now = clock.now();
            f->current_frame_microseconds = static_cast<std::uint64_t>(now);
            remaining = target_s + us_to_float(now - prev) / -1000000.0f;
        }
    }
    if (f->force_no_sleep) f->frame_behind = 1;
    const std::uint32_t idx = (f->frame_history_index + 1) & 0x1f;
    f->frame_history_index = idx;
    const std::int64_t frame_us = static_cast<std::int64_t>(f->current_frame_microseconds - f->previous_frame_microseconds);
    f->frame_history[idx].elapsed_microseconds = static_cast<std::uint64_t>(frame_us);
    f->frame_history[idx].delayed_or_behind = f->frame_behind;
    const float frame_s = us_to_float(frame_us) / 1000000.0f;
    f->measured_frame_seconds = frame_s;
    // On time, unless this frame and the late ones before it overran their
    // budget (plus 1% of the target), in which case only when the allowance
    // is used up.
    bool exhausted = allowance == 0;
    if (f->frame_behind) exhausted = walked >= allowance;
    const std::int64_t tolerance_us = seconds_to_us(f->target_frame_seconds * 0.00999999978f);
    const std::int64_t target_us = seconds_to_us(f->target_frame_seconds);
    const std::uint64_t budget = static_cast<std::uint64_t>(walked + 1) * static_cast<std::uint64_t>(target_us) + static_cast<std::uint64_t>(tolerance_us);
    bool on_time = true;
    if (static_cast<std::uint64_t>(frame_us) + behind_us >= budget) on_time = exhausted;
    if (f->force_no_sleep) on_time = false;
    f->frame_behind = on_time ? 0 : 1;
    if (!on_time) f->frame_advance_count = 0;
    float sum = 0.0f;
    for (int k = 0; k < 15; ++k) {
        const float v = f->fps_frame_time_history[k + 1];
        f->fps_frame_time_history[k] = v;
        sum = sum + v;
    }
    f->fps_frame_time_history[15] = frame_s;
    sum = sum + frame_s;
    f->calculated_fps = sum <= 0.00100000005f ? 0.0f : 16.0f / sum;
    f->force_no_sleep = 0;
    if (g_slide) {
        const auto window = get<std::uint64_t>(g_slide + (kWindowObjectSlot - kPreferredGuestSlide));
        if (window && f->request_window_60_hz) {
            put<std::uint8_t>(window + 0x4c, 1);
            put<std::uint8_t>(window + 0x4d, 1);
            put<std::uint32_t>(window + 0x50, 60);
            f->request_window_60_hz = 0;
        }
    }
}

// --- the entry ----------------------------------------------------------------

enum class Source { Guest, Ours, Compare };
Source g_source = Source::Ours;

// Compare mode: the guest's update runs; its caller's return address is
// swapped for frame_ret (asm below), which calls frame_after as the update
// returns. There ours replays the frame from the snapshot taken at entry,
// with the release time the guest reached, and every field the update
// writes is compared before any other code can touch the object.
struct Pending {
    SprjFlipper* flipper;
    std::uint64_t ret;  // the caller's return address
    SprjFlipper before;
};
thread_local Pending t_pending;
std::atomic<std::uint64_t> g_compared{0}, g_mismatched{0};

void compare_fields(const SprjFlipper& ours, const SprjFlipper& guest) {
    static int logged = 0;
    const auto miss = [&](const char* what, double a, double b) {
        g_mismatched.fetch_add(1, std::memory_order_relaxed);
        if (logged++ < 20) {
            const SprjFlipper& b0 = t_pending.before;
            host_log("frame source: compare mismatch %s: ours %g guest %g (before: mode %u/%u active %u fg %u bg %u ovr %d/%d flags %u %u %u %u reset30 %u)",
                     what, a, b, b0.primary_flip_mode_raw, b0.secondary_flip_mode_raw, b0.secondary_mode_active,
                     b0.foreground_history_count, b0.background_history_count, b0.override_foreground_history_count,
                     b0.override_background_history_count, b0.frame_behind, b0.reset_frame_history, b0.force_frame_advance,
                     b0.force_no_sleep, b0.reset_to_30_fps_pending);
        }
    };
    const auto eq = [&](const char* what, auto a, auto b) {
        if (!(a == b)) miss(what, static_cast<double>(a), static_cast<double>(b));
    };
    eq("frame_advance_count", ours.frame_advance_count, guest.frame_advance_count);
    eq("allow_frame_skip", ours.allow_frame_skip, guest.allow_frame_skip);
    eq("target_frame_seconds", ours.target_frame_seconds, guest.target_frame_seconds);
    eq("previous_frame_microseconds", ours.previous_frame_microseconds, guest.previous_frame_microseconds);
    eq("current_frame_microseconds", ours.current_frame_microseconds, guest.current_frame_microseconds);
    eq("frame_history_index", ours.frame_history_index, guest.frame_history_index);
    const std::uint32_t i = guest.frame_history_index & 0x1f;
    eq("history.elapsed", ours.frame_history[i].elapsed_microseconds, guest.frame_history[i].elapsed_microseconds);
    eq("history.behind", ours.frame_history[i].delayed_or_behind, guest.frame_history[i].delayed_or_behind);
    eq("measured_frame_seconds", ours.measured_frame_seconds, guest.measured_frame_seconds);
    eq("foreground_history_count", ours.foreground_history_count, guest.foreground_history_count);
    eq("background_history_count", ours.background_history_count, guest.background_history_count);
    eq("frame_behind", ours.frame_behind, guest.frame_behind);
    eq("reset_frame_history", ours.reset_frame_history, guest.reset_frame_history);
    eq("force_no_sleep", ours.force_no_sleep, guest.force_no_sleep);
    eq("secondary_mode_active", ours.secondary_mode_active, guest.secondary_mode_active);
    eq("secondary_mode_requested", ours.secondary_mode_requested, guest.secondary_mode_requested);
    for (int k = 0; k < 16; ++k) eq("fps_history", ours.fps_frame_time_history[k], guest.fps_frame_time_history[k]);
    eq("calculated_fps", ours.calculated_fps, guest.calculated_fps);
    eq("reset_to_30_fps_pending", ours.reset_to_30_fps_pending, guest.reset_to_30_fps_pending);
    eq("request_window_60_hz", ours.request_window_60_hz, guest.request_window_60_hz);
}

}  // namespace
extern "C" void frame_ret();
extern "C" GUEST_ABI std::uint64_t frame_after();
namespace {

// BBHOST_FRAME_STATS: the main loop's work in the frame that is ending. The
// manager's +0x28 still holds the time (gettimeofday, us) it released the last
// frame at, until the update about to run replaces it.
void note_main_frame_work(std::uint64_t m) {
    if (!frame_stats_enabled()) return;
    // The same clock the guest's gettimeofday reads (hle/libc.cpp), on both
    // platforms.
    const std::int64_t now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    const std::int64_t last = get<std::int64_t>(m + 0x28);
    if (last > 0 && now >= last && now - last < 10000000) frame_stats_main_frame(static_cast<std::uint64_t>(now - last), hle_video_flip_count());
}

// rdi: the frame-time manager.
GUEST_ABI std::int64_t frame_manager_hook(std::uint64_t, const std::uint64_t* saved) {
    np_test_tick();  // the online test hook wants the main thread, once a frame
    params_tick();   // the param repository walk (engine/params.h)
    lua_events_tick();  // named Lua events queued from any thread (engine/lua_events.h)
    event_flags_tick(); // the one-time check of our flag reader against the game's (engine/event_flags.h)
    player_data_tick(); // the loaded character's numbers, once (engine/player_data.h)
    plugins_frame();    // the plugins' on_frame callbacks (include/bbhost_plugin.h, version 2)
    change_appearance_tick();  // the mirror's editor in character creation's frame (engine/change_appearance.h)
    rebirth_tick();     // the altar's rebirth: the host's half of its talk script (engine/rebirth.h)
    world_chr_tick();   // a test warp a plugin queued this frame (BBHOST_TEST_WARP, engine/world_chr.h)
    loading_tick(hle_video_flip_count());  // the loading screens' lengths and the first in-game frame (engine/loading.h)
    auto* f = reinterpret_cast<SprjFlipper*>(static_cast<std::uintptr_t>(saved[5]));
    const bool sixty = g_game_fps == 60;
    if (g_source == Source::Ours) {
        note_main_frame_work(saved[5]);
        RealClock clock;
        g_fast_target = sixty && (g_bench_uncapped || loading_fast_now());
        sprj_flipper_update(f, clock, sixty);
        g_fast_target = false;
        if (g_frames.fetch_add(1, std::memory_order_relaxed) == 0) {
            host_log("frame source: SprjFlipper::Update is ours (0x%llx, %s); the guest's no longer runs",
                     static_cast<unsigned long long>(saved[5]), sixty ? "60" : "the game's own mode");
        }
        return 1;  // done: the stub returns to the caller
    }
    if (g_source == Source::Compare) {
        // The snapshot is taken after the hook's own writes below (at 60 it
        // puts the guest in mode 2 and clears the one-shots), so both sides
        // see the same inputs; frame_after replays it as the guest returns.
        t_pending.flipper = f;
        t_pending.ret = saved[6];
        const_cast<std::uint64_t*>(saved)[6] = reinterpret_cast<std::uint64_t>(&frame_ret);
    }
    if (g_game_fps != 60) {
        if (g_source == Source::Compare) t_pending.before = *f;
        return 0;
    }
    const std::uint64_t m = saved[5];
    note_main_frame_work(m);
    if (g_frames.fetch_add(1, std::memory_order_relaxed) == 0) {
        host_log("frame rate: the frame-time manager is 0x%llx, mode %u (one-shot %u), interval %g s",
                 static_cast<unsigned long long>(m), get<std::uint32_t>(m + 8), get<std::uint32_t>(m + 0xc), get<float>(m + 0x18));
    }
    put<std::uint32_t>(m + 8, kModeSixty);
    put<std::uint32_t>(m + 0xc, kModeSixty);
    put<std::int32_t>(m + 0x2bc, -1);
    put<std::int32_t>(m + 0x2c0, -1);
    put<std::uint8_t>(m + 0x271, 0);
    put<std::uint8_t>(m + 0x272, 0);
    put<std::uint8_t>(m + 0x2c4, 0);
    if (g_source == Source::Compare) t_pending.before = *f;
    return 0;
}

// The Frame cap option as the game started with it: 60 or more (or Off) is
// the game at 60, 30 its own pace. BBHOST_GAME_FPS=30 or 60 overrides it.
int wanted_fps() {
    if (const char* e = std::getenv("BBHOST_GAME_FPS"); e && *e) return std::atoi(e) >= 60 ? 60 : 30;
    const int cap = host_settings().frame_cap;
    return cap == 0 || cap >= 60 ? 60 : 30;
}


// Checks every site, then writes them all; false (nothing written) on any
// mismatch.
bool apply_sixty(ElfImage* image) {
    const auto va = [&](std::uint32_t bn) { return image->mem.slide + (bn - kPreferredGuestSlide); };
    for (const Constant& c : kSixtyConstants) {
        if (get<std::uint32_t>(va(c.at)) != c.was) {
            host_log("frame rate: refused, 0x%x holds 0x%08x, not 0x%08x", c.at, get<std::uint32_t>(va(c.at)), c.was);
            return false;
        }
    }
    for (const Bytes& b : kSixtyBytes) {
        if (std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va(b.at))), b.was, b.n) != 0) {
            host_log("frame rate: refused, 0x%x does not hold the expected %u bytes", b.at, b.n);
            return false;
        }
    }
    // The 1.09 image's executable segment ends at 0x54d96dc (its first PT_LOAD);
    // a site past it is in the read-write data segment, and its page goes back
    // to read-write - the frame-time slot (0x575a908) is written every frame.
    constexpr std::uint32_t kCodeEnd = 0x54d96dc;
    const auto write = [&](std::uint32_t bn, const void* src, std::size_t n) {
        const std::uint64_t at = va(bn), lo = at & ~0xfffull, hi = (at + n + 0xfff) & ~0xfffull;
        guest_protect_rwx(&image->mem, lo, hi - lo);
        std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(at)), src, n);
        if (bn < kCodeEnd) guest_protect_rx(&image->mem, lo, hi - lo);
        else guest_protect_rw(&image->mem, lo, hi - lo);
    };
    for (const Constant& c : kSixtyConstants) write(c.at, &c.now, 4);
    for (const Bytes& b : kSixtyBytes) write(b.at, b.now, b.n);
    // The code rewrites: all of them or none, and the constants stand either way.
    static const bool sites_on = [] {
        const char* e = std::getenv("BBHOST_SIXTY_SITES");
        return !(e && e[0] == '0');
    }();
    if (!sites_on) {
        host_log("frame rate: BBHOST_SIXTY_SITES=0, the 60 FPS++ code rewrites left out");
        return true;
    }
    for (const Site& site : kSixtySites) {
        const std::vector<std::uint8_t> was = hex_bytes(site.was);
        if (std::memcmp(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(va(site.at))), was.data(), was.size()) != 0) {
            host_log("frame rate: 0x%x does not hold the 60 FPS++ list's old bytes; its %zu code rewrites left out", site.at,
                     sizeof(kSixtySites) / sizeof(kSixtySites[0]));
            return true;
        }
    }
    for (const Site& site : kSixtySites) {
        const std::vector<std::uint8_t> now = hex_bytes(site.now);
        write(site.at, now.data(), now.size());
    }
    g_sites_applied = sizeof(kSixtySites) / sizeof(kSixtySites[0]);
    return true;
}

}  // namespace

// As the guest's update returns (compare mode): ours on the snapshot, the
// fields compared, then on to the caller.
extern "C" GUEST_ABI std::uint64_t frame_after() {
    Pending& p = t_pending;
    SprjFlipper replay = p.before;
    ReplayClock clock(static_cast<std::int64_t>(p.flipper->current_frame_microseconds));
    const std::uint64_t slide = g_slide;
    g_slide = 0;  // the window request is not replayed (it would be written twice)
    sprj_flipper_update(&replay, clock, g_game_fps == 60);
    g_slide = slide;
    replay.request_window_60_hz = p.flipper->request_window_60_hz;
    compare_fields(replay, *p.flipper);
    const std::uint64_t n = g_compared.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n % 1800 == 0 || n == 60) {
        host_log("frame source: compare %llu frames, %llu field mismatches", static_cast<unsigned long long>(n),
                 static_cast<unsigned long long>(g_mismatched.load()));
    }
    return p.ret;
}
extern "C" void* g_frame_after_thunk = nullptr;
#if defined(_WIN32)
#define FR_TYPE_(n)
#else
#define FR_TYPE_(n) ".type " #n ", @function\n"
#endif
asm(".text\n.globl frame_ret\n" FR_TYPE_(frame_ret) "frame_ret:\n"
    "push %rax\n push %rdx\n call *g_frame_after_thunk(%rip)\n"
    "mov %rax, %r11\n pop %rdx\n pop %rax\n jmp *%r11\n");

int frame_rate_game_fps() { return g_game_fps; }

void frame_rate_install(ElfImage* image) {
    const std::uint64_t at = image->mem.slide + (kFrameManager - kPreferredGuestSlide);
    g_slide = image->mem.slide;
    loading_bind(image->mem.slide);
    // BBHOST_FRAME_SOURCE: 1 (default) SprjFlipper::Update is ours; 0 the
    // guest's runs with the mode hook; 2 compare, the guest's runs and ours
    // replays each frame beside it.
    if (const char* e = std::getenv("BBHOST_FRAME_SOURCE"); e && e[0]) {
        g_source = e[0] == '0' ? Source::Guest : e[0] == '2' ? Source::Compare : Source::Ours;
    }
    if (g_source == Source::Compare) g_frame_after_thunk = hle_wrap_fn(reinterpret_cast<void*>(&frame_after));
    if (wanted_fps() < 60) {
        host_log("frame rate: the game's own 30");
        // Ours runs the game's own mode at 30 (or the guest's does with
        // BBHOST_FRAME_SOURCE=0, when only the online test tick needs the hook).
        {  // ours runs the game's own mode at 30 (the param walk and the online tick ride on the same hook)
            if (!engine_prologue_hook(image, at, kFrameManagerPrologue, sizeof(kFrameManagerPrologue),
                                      reinterpret_cast<void*>(&frame_manager_hook))) {
                host_log("frame source: refused, 0x%llx is not the frame-time manager's prologue; the guest's runs",
                         static_cast<unsigned long long>(kFrameManager));
                g_source = Source::Guest;
            }
        }
        return;
    }
    if (!engine_prologue_hook(image, at, kFrameManagerPrologue, sizeof(kFrameManagerPrologue),
                              reinterpret_cast<void*>(&frame_manager_hook))) {
        host_log("frame rate: refused, 0x%llx is not the frame-time manager's prologue; the game stays at 30",
                 static_cast<unsigned long long>(kFrameManager));
        return;
    }
    if (!apply_sixty(image)) {
        host_log("frame rate: the game's steps could not be converted; staying at 30");
        return;  // the hook keeps mode 2 only while g_game_fps says 60
    }
    g_game_fps = 60;
    hle_video_set_fps_cap(host_settings().frame_cap);  // at least 60 now (hle/video.cpp)
    host_log("frame rate: 60 (the frame-time manager's mode 2; %zu constants and %zu other values converted, %zu 60 FPS++ "
             "code sites)",
             sizeof(kSixtyConstants) / sizeof(kSixtyConstants[0]), sizeof(kSixtyBytes) / sizeof(kSixtyBytes[0]),
             g_sites_applied);
}

void frame_rate_set_time_scale(float scale) {
    if (!(scale >= 0.1f)) scale = 0.1f;  // also NaN
    if (scale > 4.0f) scale = 4.0f;
    g_time_scale.store(scale, std::memory_order_relaxed);
    // An eased change sets it every frame: logged in steps of a tenth.
    static float logged = 1.0f;
    if (std::fabs(scale - logged) >= 0.1f || (scale == 1.0f && logged != 1.0f)) {
        logged = scale;
        host_log("frame rate: time scale %.2f", static_cast<double>(scale));
    }
}
