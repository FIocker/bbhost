#pragma once

// The mouse turns the camera the way Dark Souls III PC does (docs/running.md,
// "Mouse"): not through the right stick, but as an angle per count of the
// mouse's own movement, added to the follow camera's pitch and yaw once a
// frame. DS3 reads the DirectInput counts and turns its ChrExFollowCam by
//
//     0.2 * (sensitivity * 0.15 + 0.5) degrees a count
//
// (0.25 at its default 5), after the stick has turned it and only when the
// stick has not - with no frame time in it, no dead zone, no cap and no
// acceleration, so the same movement always turns the same angle. Bloodborne
// has the same camera class without the mouse; this adds DS3's step where DS3
// has it, in NS_SPRJ::ChrExFollowCam::Update. Locked on, DS3's mouse does not
// turn the camera but switches the target on a flick, which this does through
// the right stick the game already switches with.

#include <cstdint>

struct ElfImage;

void mouse_camera_install(ElfImage* image);
bool mouse_camera_installed();

// DS3's degrees per count at a sensitivity of 0..10.
float mouse_camera_degrees_per_count(int sensitivity);

// For the pad read: whether the mouse has just flicked for a lock-on target
// switch, and the right stick that switches it (-1..1, x right, y down).
bool mouse_camera_flick(float& x, float& y);

// The exit report: how the mouse turned the camera.
void mouse_camera_report();
