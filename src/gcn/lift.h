// Typed per-pixel lifting of GCN pixel shaders.
//
// The translator (translate.h) emulates the SIMD machine: registers are u32
// variables, EXEC and VCC are lane masks rebuilt with subgroup ballots, and
// every masked write selects between reloaded words. The lifter emits the
// same computation for one pixel as typed SSA values: floats stay floats, a
// lane mask is the pixel's own boolean, and the EXEC save/branch/restore idiom
// becomes selects over values computed in uniform control flow.
//
// It lifts only what it can show is the same computation and rejects anything
// else with a reason; the caller then keeps the translated shader. The facts
// each lift relied on are returned in `proof`.
#pragma once

#include "gcn/isa.h"
#include "gcn/translate.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gcn {

// The GCN program's wave and quad operations and what the lift made of them.
// Speed-only: shortcuts a wave takes that change no pixel's result - the lift
// runs per pixel without them, and an HLSL emitter drops them too. Semantic:
// operations whose results depend on the other pixels of a quad or on which
// pixels survive - kept, as Shader Model 6 quad operations, derivatives,
// helper-lane queries or discard.
struct WaveOps {
    // speed-only (dropped)
    std::uint32_t execz_skips = 0;   // s_cbranch_execz over a region: the region runs, its writes masked per pixel
    std::uint32_t kill_skips = 0;    // s_cbranch_scc0 over a kill region: the pixel's own guard bit masks it
    std::uint32_t wave_tests = 0;    // s_cbranch_vccz/vccnz on a VCC shown the same in every lane: a per-pixel condition
    std::uint32_t lane_exits = 0;    // a divergent loop's per-lane exit idiom (s_andn2_b64 + s_cbranch_scc0): a per-pixel break
    std::uint32_t whole_quad = 0;    // s_wqm_b64: helper lanes run whole quads anyway
    std::uint32_t spill_lanes = 0;   // v_writelane_b32 / v_readlane_b32 of a spilled scalar: a register
    // semantic (kept)
    std::uint32_t quad_swizzles = 0;  // ds_swizzle_b32 within a quad: QuadReadLaneAt / QuadReadAcross*
    std::uint32_t implicit_lod = 0;   // samples whose LOD comes from the quad's derivatives
    std::uint32_t lod_queries = 0;    // image_get_lod: CalculateLevelOfDetailUnclamped
    std::uint32_t helper_queries = 0; // whether a quad holds a live pixel (HelperInvocation + quad reads)
    std::uint32_t discards = 0;       // pixels discarded: EXEC clear at s_endpgm, or a clip distance below 0
};

struct LiftResult {
    std::vector<std::uint32_t> spirv;
    std::vector<std::string> rejections;  // why the program was not lifted; empty on success
    std::vector<std::string> proof;       // what the lift relied on
    std::string listing;                  // the program with the typed value each write produced
    WaveOps wave;                         // on success
    bool ok() const { return rejections.empty() && !spirv.empty(); }
};

// Lifts a pixel shader. `reference` is the translator's result for the same
// program and options, with cb_no_fallback set: the lifted module declares the
// reference's descriptor bindings, inputs and outputs, so it runs with the
// descriptor sets written for the reference pipeline.
LiftResult lift_pixel_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference);

// Lifts a vertex shader on Vulkan vertex input (TranslateOptions::vertex_input,
// no inlined fetch shader) under the same rules: inputs at s_swappc_b64, each
// element converted as the translator converts it - by its formats as
// constants, or with vertex_formats_from_params by its descriptor in the params
// block or the relink's specialization constant -, the position and params
// exported with EXEC known set, samples at level 0.
LiftResult lift_vertex_shader(const Program& program, const TranslateOptions& options, const TranslateResult& reference);

}  // namespace gcn
