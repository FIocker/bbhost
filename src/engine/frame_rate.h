#pragma once

#include <string>

// The game's own frame pace: 30, 60, 90 or uncapped.
//
// FD4's frame-time manager (0x2434770, called once a frame from the main
// loop) picks an interval by mode - the int at +8, or +0xc while the one-shot
// flag +0x276 is set - through a switch at 0x24347f5. Modes 0, 1, 3 and 4 wait
// out 1/30 s; mode 2 waits 1/60 with a vblank interval of 1. The engine has a
// 60 fps mode; the game never selects it. The wait is a usleep to within 5 ms
// of the deadline and then a spin on gettimeofday - which is why the main
// thread reads ~90% busy at 30 fps while half of that is waiting. Ours
// replaces it (SprjFlipper::Update, BBHOST_FRAME_SOURCE).
//
// The Frame rate option (or BBHOST_GAME_FPS=30/60/90/0) when the game starts:
//  - 30: the game's own.
//  - 60: mode 2, with the game's fixed steps converted from Kyo's "60 FPS++"
//    list (frame_rate.cpp's tables and engine/sixty_fps_sites.inc).
//  - 90: the same with a 1/90 s target, the rate-dependent values at 90 and
//    Kyo's "90 FPS++" additions (the cloth package: engine/ninety_fps_sites.inc).
//  - uncapped: the 90 package, a target of the display's refresh rate with
//    V-Sync on and 1/240 s with it off (BBHOST_UNCAPPED_FPS=N sets it), and
//    the values that depend on the frame's length written every frame from
//    the measured frame time - the game's own step already follows it through
//    the FPS++ frame-time cave (0x2418e3d), clamped to [target, 1/30].
// At 90 and uncapped flips complete as soon as they are recorded (the game
// paces itself; hle/video.cpp). The one-shot overrides (+0x2bc, +0x2c0,
// +0x271, +0x272, +0x2c4) are cleared so nothing drops the game back to 30.
//
// The choice is made when the game starts: the game's fixed 1/30 s steps are
// constants in its code, and a rate needs them all changed together.

struct ElfImage;

void frame_rate_install(ElfImage* image);

// The pace the game was started at: 30, 60, 90, or 0 for uncapped.
int frame_rate_game_fps();

// Game time against real time (plugins' set_time_scale): 0.5 is half speed,
// 2 double (as far as the machine keeps up), clamped to 0.1-4. The game steps
// a fixed amount a frame, so this is the frame pacing's interval. (Uncapped,
// the step follows the frame time, so only the fixed steps slow down.)
void frame_rate_set_time_scale(float scale);

// The limiter since the last call, for the 300-flip report's timing line:
// its margin (learnt from its own sleeps' lateness, or
// BBHOST_LIMITER_MARGIN_US), how late its sleeps woke, and how long it spun
// a frame.
std::string frame_rate_limiter_window();
