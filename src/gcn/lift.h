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

struct LiftResult {
    std::vector<std::uint32_t> spirv;
    std::vector<std::string> rejections;  // why the program was not lifted; empty on success
    std::vector<std::string> proof;       // what the lift relied on
    std::string listing;                  // the program with the typed value each write produced
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
