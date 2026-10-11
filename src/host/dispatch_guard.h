#pragma once

// Indirect dispatches that ask for more thread groups than a native renderer
// of this engine ever would (host_gpu_dispatch skips them), and the command
// processor's recent small copies, which name where such a count came from.
//
// D3D11 caps a dispatch at 65,535 thread groups a dimension
// (D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION), and DS3 PC - this
// engine's own PC renderer - keeps within it, so a count past it is not the
// game's. In bbhost it is an append buffer's counter that nothing wrote: GX
// keeps those counters in GDS around each compute dispatch (DMA_DATA from
// memory or a starting value into GDS before, out of GDS into memory after),
// which bbhost does not carry out yet, so a counter keeps whatever its memory
// held before - and GX copies it into the next dispatch's arguments (a 4-byte
// DMA_DATA, sub_2568be0). At the Forbidden Woods a new particle system's
// counter held float bits, 0xbd4ede96, and the particle simulation
// (CS_VFX_StandardParticle) asked for 3,176,062,614 groups with every thread
// live: the GPU ran it for minutes.

#include <cstdint>

namespace dispatch_guard {

constexpr std::uint32_t kMaxGroups = 65535;

inline bool groups_fit(const std::uint32_t groups[3]) {
    return groups[0] <= kMaxGroups && groups[1] <= kMaxGroups && groups[2] <= kMaxGroups;
}

// The last memory-to-memory copies of 16 bytes or less the command processor
// made: GX fills an indirect dispatch's arguments with one.
class SmallCopies {
public:
    void note(std::uint64_t dst, std::uint64_t src, std::uint64_t bytes) {
        if (bytes == 0 || bytes > 16) return;
        ring_[next_++ % kSlots] = Copy{dst, src, bytes};
    }
    // Where the newest copy that covered `va` read it from.
    bool source_of(std::uint64_t va, std::uint64_t& src) const {
        for (std::uint64_t i = 0; i < kSlots && i < next_; ++i) {
            const Copy& c = ring_[(next_ - 1 - i) % kSlots];
            if (va >= c.dst && va < c.dst + c.bytes) {
                src = c.src + (va - c.dst);
                return true;
            }
        }
        return false;
    }

private:
    struct Copy {
        std::uint64_t dst = 0, src = 0, bytes = 0;
    };
    static constexpr std::uint64_t kSlots = 32;
    Copy ring_[kSlots];
    std::uint64_t next_ = 0;
};

}  // namespace dispatch_guard
