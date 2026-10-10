// gcnlift: lift a captured draw's pixel or vertex shader.
//
//   gcnlift <capture-dir> <out-dir> [--vs]
//
// Rebuilds the stage's TranslateOptions the renderer used from the capture
// manifest, translates the captured GCN words again and requires the result to
// equal the captured SPIR-V word for word (so the options are the real ones),
// then lifts the program against that reference. Writes lifted.spv (validated),
// listing.txt and report.json with the proof or the rejections.
// Replay the result with `drawreplay <capture-dir> --ps|--vs <out-dir>/lifted.spv`.
// A vertex shader lifts on vertex input only (captures record it).
// Exit 0: lifted and valid; 1: rejected or invalid; 2: bad input.
#include "gcn/isa.h"
#include "gcn/lift.h"
#include "gcn/translate.h"
#include "replay/json.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error(why); }

std::vector<std::uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) fail("cannot read " + p.string());
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_file(const fs::path& p, const std::string& text) {
    std::ofstream f(p, std::ios::binary);
    if (!f || !f.write(text.data(), static_cast<std::streamsize>(text.size()))) fail("cannot write " + p.string());
}

std::vector<std::uint32_t> words_of(const std::vector<std::uint8_t>& b, const std::string& what) {
    if (b.size() % 4) fail(what + " is not a whole number of dwords");
    std::vector<std::uint32_t> w(b.size() / 4);
    std::memcpy(w.data(), b.data(), b.size());
    return w;
}

std::string validate(const std::vector<std::uint32_t>& spirv) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string msg;
    tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& pos, const char* message) {
        if (msg.empty()) msg = "word " + std::to_string(pos.index) + ": " + message;
    });
    spvtools::ValidatorOptions vo;
    vo.SetAllowOffsetTextureOperand(true);
    return tools.Validate(spirv.data(), spirv.size(), vo) ? std::string() : (msg.empty() ? "invalid" : msg);
}

std::uint32_t word_at(const json::Value& list, std::size_t k, const char* what) {
    if (list.type != json::Value::Type::Array || k >= list.array.size()) fail(std::string("manifest ") + what + " is too short");
    return static_cast<std::uint32_t>(json::as_u64(list.array[k], what));
}

// The draw's clip varying (TranslateOptions::kVsOutCntlClipVarying): the pixel
// shader's ps_clip_discard, 0 in captures from before it was recorded.
std::uint32_t clip_discard(const json::Value& tr) {
    const json::Value* c = tr.find("ps_clip_discard");
    return c ? static_cast<std::uint32_t>(json::as_u64(*c, "ps_clip_discard")) : 0;
}

// The pixel-stage options render.cpp's gfx_pipeline() builds, from what the
// capture recorded about the draw.
gcn::TranslateOptions pixel_options(const json::Value& m) {
    const json::Value& reg = json::member(m, "registers");
    const json::Value& tr = json::member(m, "translate");
    const json::Value& ps = json::arr(m, "stages").at(1);
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Pixel;
    o.rsrc1 = word_at(json::member(reg, "ps_pgm"), 2, "ps_pgm");
    o.rsrc2 = word_at(json::member(reg, "ps_pgm"), 3, "ps_pgm");
    o.ps_input_ena = static_cast<std::uint32_t>(json::u64(reg, "ps_input_ena"));
    o.descriptor_set = 1;
    for (const json::Value& b : json::arr(ps, "images")) {
        // The renderer asks for a cube; the binding records the 2D array it became.
        const json::Value* cube = b.find("cube");
        const bool is_cube = cube && cube->type == json::Value::Type::Bool && cube->boolean;
        o.image_dims.push_back(is_cube ? std::pair<std::uint32_t, bool>{3, false}
                                       : std::pair<std::uint32_t, bool>{json::u32(b, "dim"), json::flag(b, "arrayed")});
    }
    for (const json::Value& b : json::arr(ps, "samplers")) {
        o.sampler_force_unnormalized.push_back((word_at(json::member(b, "ssharp"), 0, "ssharp") >> 15) & 1);
    }
    o.cb_ssbo = json::flag(tr, "cb_ssbo");
    o.cb_no_fallback = json::flag(tr, "cb_no_fallback");
    o.exec_known = json::flag(tr, "exec_known");
    o.early_fragment_tests = json::flag(tr, "early_fragment_tests");
    o.debug_ps = static_cast<int>(json::u64(tr, "debug_ps"));
    const json::Value& cntl = json::member(reg, "ps_input_cntl");
    for (std::size_t k = 0; k < 32; ++k) {
        const std::uint32_t c = word_at(cntl, k, "ps_input_cntl");
        if ((c >> 10) & 1) o.ps_flat_mask |= 1u << k;
        // Input k is read at the location of the register the input map names
        // (render.cpp ps_input_locations); the vertex shader writes register r at r.
        o.ps_input_map.push_back(static_cast<std::uint8_t>(c & 0x1f));
    }
    o.ps_clip_discard = clip_discard(tr);
    return o;
}

// The vertex-stage options gfx_pipeline() builds. Captures before vs_out_cntl
// and vs_invariant were recorded default them (0, off).
gcn::TranslateOptions vertex_options(const json::Value& m) {
    const json::Value& reg = json::member(m, "registers");
    const json::Value& tr = json::member(m, "translate");
    const json::Value& vs = json::arr(m, "stages").at(0);
    gcn::TranslateOptions o;
    o.stage = gcn::Stage::Vertex;
    o.rsrc1 = word_at(json::member(reg, "vs_pgm"), 2, "vs_pgm");
    o.rsrc2 = word_at(json::member(reg, "vs_pgm"), 3, "vs_pgm");
    if (const json::Value* cntl = tr.find("vs_out_cntl")) o.vs_out_cntl = static_cast<std::uint32_t>(json::as_u64(*cntl, "vs_out_cntl"));
    // build_gfx_pipeline adds the clip varying's host bits to the register's.
    if (const std::uint32_t clip = clip_discard(tr)) {
        o.vs_out_cntl |= gcn::TranslateOptions::kVsOutCntlClipVarying | ((clip >> 8) & 31) << gcn::TranslateOptions::kVsOutCntlClipLocationShift;
    }
    const json::Value* inv = tr.find("vs_invariant");
    o.invariant_position = inv && inv->type == json::Value::Type::Bool && inv->boolean;
    o.descriptor_set = 0;
    for (const json::Value& b : json::arr(vs, "images")) {
        const json::Value* cube = b.find("cube");
        const bool is_cube = cube && cube->type == json::Value::Type::Bool && cube->boolean;
        o.image_dims.push_back(is_cube ? std::pair<std::uint32_t, bool>{3, false}
                                       : std::pair<std::uint32_t, bool>{json::u32(b, "dim"), json::flag(b, "arrayed")});
    }
    for (const json::Value& b : json::arr(vs, "samplers")) {
        o.sampler_force_unnormalized.push_back((word_at(json::member(b, "ssharp"), 0, "ssharp") >> 15) & 1);
    }
    o.cb_ssbo = json::flag(tr, "cb_ssbo");
    o.cb_no_fallback = json::flag(tr, "cb_no_fallback");
    o.exec_known = json::flag(tr, "exec_known");
    const json::Value* vi = m.find("vertex_input");
    if (vi && vi->type == json::Value::Type::Object) {
        for (const json::Value& e : json::arr(*vi, "elements")) {
            gcn::VertexElement el;
            el.location = json::u32(e, "location");
            el.vdata = json::u32(e, "vdata");
            el.count = json::u32(e, "count");
            el.w3 = static_cast<std::uint32_t>(json::u64(e, "w3"));
            o.vertex_input.push_back(el);
        }
        // As the capture records it. Captures from before step 6c do not, and are
        // rebuilt with formats from params wherever the loader supports them
        // (drawreplay fills StageParams::vertex_formats from these elements).
        if (const json::Value* p = tr.find("vertex_formats_from_params"); p && p->type == json::Value::Type::Bool) {
            o.vertex_formats_from_params = p->boolean;
        } else {
            o.vertex_formats_from_params = gcn::vertex_formats_from_params_supported(o.vertex_input);
        }
    }
    return o;
}

int run(const fs::path& capture, const fs::path& out, bool vs) {
    const std::vector<std::uint8_t> text = read_file(capture / "manifest.json");
    json::Value m;
    std::string error;
    if (!json::parse(std::string(text.begin(), text.end()), m, error)) fail("manifest.json: " + error);
    if (json::str(m, "schema") != "bbhost-draw-capture") fail("not a draw capture");
    const json::Value& blobs = json::member(m, "blobs");
    const auto blob = [&](const std::string& name) { return read_file(capture / json::str(json::member(blobs, name.c_str()), "file")); };
    const json::Value& stage = json::arr(m, "stages").at(vs ? 0 : 1);
    if (!json::flag(stage, "present")) fail(vs ? "the capture has no vertex shader" : "the capture has no pixel shader");
    const std::vector<std::uint32_t> gcn_words = words_of(blob(json::str(stage, "gcn")), vs ? "vs-gcn" : "ps-gcn");
    const std::vector<std::uint32_t> captured = words_of(blob(json::str(stage, "spirv")), vs ? "vs-spirv" : "ps-spirv");

    const gcn::Program program = gcn::decode(gcn_words.data(), gcn_words.size());
    gcn::TranslateOptions options = vs ? vertex_options(m) : pixel_options(m);
    if (vs) {
        // Linking by register (render.cpp register_links): the vertex shader
        // writes param register r at location r, whatever older captures recorded.
        options.link_outputs = true;
        for (std::uint8_t r = 0; r < 32; ++r) options.output_links.push_back(r);
    }
    gcn::Program fetch_program;  // a vertex shader without vertex input inlines its fetch shader
    if (vs && options.vertex_input.empty()) {
        const json::Value* fetch = json::member(m, "shaders").find("fetch_gcn");
        if (fetch && fetch->type == json::Value::Type::String) {
            const std::vector<std::uint32_t> fetch_words = words_of(blob(fetch->string), "fetch-gcn");
            fetch_program = gcn::decode(fetch_words.data(), fetch_words.size());
            options.fetch = &fetch_program;
        }
    }
    const gcn::TranslateResult reference = gcn::translate(program, options);
    const gcn::LiftResult lifted = vs ? gcn::lift_vertex_shader(program, options, reference) : gcn::lift_pixel_shader(program, options, reference);
    // A capture taken with BBHOST_DECOMP bound the lifted stage; that is what
    // the rebuilt options have to reproduce.
    const json::Value* stage_lifted = json::member(m, "translate").find(vs ? "vs_lifted" : "ps_lifted");
    const bool captured_lifted = stage_lifted && stage_lifted->type == json::Value::Type::Bool && stage_lifted->boolean;
    // Debug names aside: the renderer puts the engine's names on a lift
    // (gcn::name_lift, from the container's reflection, which a capture does
    // not carry), and they change nothing the module computes.
    const auto unnamed = [](const std::vector<std::uint32_t>& m) {
        if (m.size() < 5) return m;
        std::vector<std::uint32_t> out(m.begin(), m.begin() + 5);
        for (std::size_t i = 5; i < m.size();) {
            const std::uint32_t n = m[i] >> 16, op = m[i] & 0xffff;
            if (!n || i + n > m.size()) return m;
            if (op != 5 && op != 6) out.insert(out.end(), m.begin() + static_cast<std::ptrdiff_t>(i), m.begin() + static_cast<std::ptrdiff_t>(i + n));  // OpName, OpMemberName
            i += n;
        }
        return out;
    };
    const bool same_reference = captured_lifted ? lifted.ok() && unnamed(lifted.spirv) == unnamed(captured)
                                                : reference.ok() && unnamed(reference.spirv) == unnamed(captured);
    const std::string invalid = lifted.ok() ? validate(lifted.spirv) : std::string();

    fs::create_directories(out);
    json::Value report = json::Value::make_object();
    report.set("schema", "bbhost-gcn-lift");
    report.set("version", 1);
    report.set("capture", capture.string());
    report.set("pipeline", json::str(json::member(m, "identity"), "pipeline"));
    report.set("stage", vs ? "vertex" : "pixel");
    report.set("reference_matches_capture", same_reference);  // the rebuilt module of the captured kind
    report.set("captured_module", captured_lifted ? "lifted" : "translated");
    report.set("lifted", lifted.ok());
    report.set("spirv_valid", lifted.ok() && invalid.empty());
    if (!invalid.empty()) report.set("spirv_error", invalid);
    json::Value proof = json::Value::make_array();
    for (const std::string& p : lifted.proof) proof.push(p);
    report.set("proof", proof);
    json::Value rejections = json::Value::make_array();
    for (const std::string& r : lifted.rejections) rejections.push(r);
    report.set("rejections", rejections);
    report.set("reference_words", static_cast<unsigned long>(reference.spirv.size()));
    report.set("lifted_words", static_cast<unsigned long>(lifted.spirv.size()));
    {
        // Step 2 of per-shader compiles: the dimensions predicted from the
        // program alone against the ones the captured draw bound.
        const std::vector<gcn::PredictedImage> predicted = gcn::predict_image_dims(program, reference);
        json::Value dims = json::Value::make_object();
        json::Value pred = json::Value::make_array(), bound = json::Value::make_array();
        bool match = true, conflict = false;
        for (std::size_t k = 0; k < predicted.size(); ++k) {
            const std::pair<std::uint32_t, bool> a = k < options.image_dims.size() ? options.image_dims[k] : std::pair<std::uint32_t, bool>{1, false};
            pred.push(std::to_string(predicted[k].dim) + (predicted[k].arrayed ? "a" : "") + (predicted[k].conflict ? "!" : ""));
            bound.push(std::to_string(a.first) + (a.second ? "a" : ""));
            match &= predicted[k].dim == a.first && predicted[k].arrayed == a.second;
            conflict |= predicted[k].conflict;
        }
        dims.set("predicted", pred);
        dims.set("bound", bound);
        dims.set("match", match);
        dims.set("conflict", conflict);
        report.set("image_dims", dims);
    }
    write_file(out / "report.json", json::dump(report));
    write_file(out / "listing.txt", lifted.listing);
    const auto spv_text = [](const std::vector<std::uint32_t>& w) { return std::string(reinterpret_cast<const char*>(w.data()), w.size() * 4); };
    if (lifted.ok()) write_file(out / "lifted.spv", spv_text(lifted.spirv));
    if (reference.ok()) write_file(out / "reference.spv", spv_text(reference.spirv));
    // Diagnostic floor for replay benches: the same translation with every
    // export magenta (BBHOST_DEBUG_PS_COLOR=1), so the rest of the pixel shader
    // is dead code while the pipeline, bindings and fragment tests stay the same.
    if (!vs) {
        gcn::TranslateOptions magenta_options = options;
        magenta_options.debug_ps = 1;
        const gcn::TranslateResult magenta = gcn::translate(program, magenta_options);
        if (magenta.ok() && validate(magenta.spirv).empty()) write_file(out / "magenta.spv", spv_text(magenta.spirv));
    }
    std::printf("rebuilt %s module %s the captured SPIR-V; lift %s%s\n", captured_lifted ? "lifted" : "translated",
                same_reference ? "matches" : "DOES NOT MATCH", lifted.ok() ? "ok" : "rejected",
                invalid.empty() ? "" : (" but INVALID: " + invalid).c_str());
    for (const std::string& r : lifted.rejections) std::printf("  rejected: %s\n", r.c_str());
    for (const std::string& p : lifted.proof) std::printf("  proof: %s\n", p.c_str());
    if (!same_reference) {
        std::printf("  the rebuilt options do not reproduce the pipeline's shader; the lift is not comparable\n");
        return 1;
    }
    return lifted.ok() && invalid.empty() ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const bool vs = argc == 4 && std::strcmp(argv[3], "--vs") == 0;
    if (argc != 3 && !vs) {
        std::fprintf(stderr, "usage: gcnlift <capture-dir> <out-dir> [--vs]\n");
        return 2;
    }
    try {
        return run(argv[1], argv[2], vs);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gcnlift: %s\n", e.what());
        return 2;
    }
}
