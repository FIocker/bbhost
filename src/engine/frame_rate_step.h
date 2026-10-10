#pragma once

// The frame rate as code that runs on the game's side of the thunks reads it:
// a decomp leaf calls nothing of the host's and touches no host TLS, so it
// reads plain globals. Written by engine/frame_rate.cpp; header-defined so an
// eboot kit test links without it (there, as at the game's own 30, nothing
// is converted).

#include <atomic>

namespace frame_rate_step {

// This frame's step at a converted rate - the measured frame time clamped to
// [target, 1/30], what the game's task step is - and 0 at the game's own 30.
inline std::atomic<float> g_frame_seconds{0.0f};
}  // namespace frame_rate_step
