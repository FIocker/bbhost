// Player data (docs/engine-map.md, "Player data and progression"): the
// funnels the player's echoes and levels pass through, and the drop chance a
// kill rolls with, as ours.
//
//   sub_1981830            item discovery: arcane through its curve, plus the
//                          effects' itemDropRate once one in effect is an
//                          item-discovery effect (stateInfo 0x42)
//   PlayerIns::vf122       echoes gained - every gain passes here: the
//                          player's soulRate effects, the 999,999,999 cap, the
//                          echoes ever gained, the HUD's counters
//   sub_1cfc860            the echoes a kill gives: the victim's haveSoulRate
//                          effects (scaled by the clear count when the effect
//                          is bGameClearBonus), x0.5 when halved, x1.2 for a
//                          cooperator, rounded up past 5e-6
//   lua_cli_ExcutePenalty  the scripts' penalty (the Lua binding ExcutePenalty):
//                          a share of the echoes and some insight taken - not
//                          what a death costs, which a session with a death
//                          showed goes elsewhere
//   sub_1f2f5e0            the level-up screen's price check: the planned
//                          level through CalcCorrectGraph row 200, against the
//                          echoes not yet committed
//
// A character's effects in effect: its holder at ChrIns +0x1c8, the first
// node at +0x8; a node's state bits at +0x1c (any of 0x800c0003: not in
// effect), its SpEffectParam row at +0x48, the next node at +0x58. The rows'
// fields by the game's PARAMDEF (SP_EFFECT_PARAM_ST): soulRate +0xe0,
// haveSoulRate +0xf4, stateInfo +0x156, bGameClearBonus (+0x160 bit 6),
// itemDropRate +0x1fc. The local player's record (GameDataMan 0x593b130,
// +0x8): insight +0x84, echoes +0x94, the echoes ever gained +0x98 (written
// as 64 bits, read as 32), the echoes an effect of stateInfo 0x72 has
// gathered +0xfc.
//
// Every constant is read where the game reads it, so a patch to one reaches
// ours too. All five are leaves: each runs on the game's thread, calls only
// the game's own code, and costs what the game's did.
#include "decomp/decomp.h"

#include "log.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <xmmintrin.h>

namespace {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;
using ull = unsigned long long;

template <class T>
T ld(u64 p) {
    return *reinterpret_cast<const T*>(static_cast<std::uintptr_t>(p));
}
template <class T>
void st(u64 p, T v) {
    *reinterpret_cast<T*>(static_cast<std::uintptr_t>(p)) = v;
}

// Where the game keeps what these read (Binary Ninja addresses).
constexpr u64 kGameDataMan = 0x593b130;  // GameDataMan*: +0x8 the local player's record, +0x68 the clear count
constexpr u64 kLuaEventMan = 0x593b0c8;  // SprjLuaEventMan*
constexpr u64 kMenuMan = 0x5962878;      // MenuMan*: +0x20c, +0x210 the HUD's echo counters
constexpr u64 kClearScale = 0x4b31ef0;   // f32[7]: haveSoulRate's scale by clear count (1 to 6: 1, 1.1, 1.25, 1.5, 2, 2.5)
constexpr u64 kParamRow = 0x2314b60;     // a CalcCorrectGraph row by id: {id, row} (0 when there is none)
constexpr u64 kKillRecord = 0x1c98290;   // what a kill tells next (the player's +0x2a0, victim, cooperator, 0)

// The constants, where each function loads them from.
constexpr u64 kDiscoveryOne = 0x4d26ac0, kDiscovery8 = 0x4d26ac4, kDiscovery30 = 0x4d26ac8, kDiscoveryTop = 0x4d26acc,
              kDiscovery50 = 0x4d26ad0, kDiscoveryMinus30 = 0x4d26ad4, kDiscoverySlope30 = 0x4d26ad8,
              kDiscoveryBase30 = 0x4d26adc, kDiscoveryMinus8 = 0x4d26ae0, kDiscoverySlope8 = 0x4d26ae4;  // 1, 8, 30, 2.1, 50, -30, 0.02, 1.7, -8, 0.0318
constexpr u64 kGainOne = 0x4d27eb8;                                                               // 1
constexpr u64 kKillOne = 0x4d27ebc, kKillHalf = 0x4d27ec0, kKillCoop = 0x4d27ec4, kKillRoundUp = 0x4d27ec8;  // 1, 0.5, 1.2, 5e-6
constexpr u64 kPriceHalf = 0x4d289c0;                                                             // 0.5

float k(u64 bn) { return ld<float>(decomp_leaf_guest(bn)); }

// The effects list (above).
constexpr u64 kFirst = 0x8, kState = 0x1c, kParam = 0x48, kNext = 0x58;
constexpr u32 kNotInEffect = 0x800c0003;
constexpr u64 kSoulRate = 0xe0, kHaveSoulRate = 0xf4, kStateInfo = 0x156, kClearBonusByte = 0x160, kItemDropRate = 0x1fc;
constexpr u8 kClearBonusBit = 0x40;
constexpr u16 kStateDiscovery = 0x42, kStateGathering = 0x72;

// The record (above).
constexpr u64 kInsight = 0x84, kEchoes = 0x94, kEchoesEver = 0x98, kEchoesGathered = 0xfc;
constexpr i32 kEchoCap = 999999999;

i32 wrap_add(i32 a, i32 b) { return static_cast<i32>(static_cast<u32>(a) + static_cast<u32>(b)); }
i32 wrap_sub(i32 a, i32 b) { return static_cast<i32>(static_cast<u32>(a) - static_cast<u32>(b)); }
// cvttss2si: out of range or NaN is 0x80000000.
i32 truncate(float f) { return _mm_cvttss_si32(_mm_set_ss(f)); }
// A sum the game keeps in [0, cap]: a wrapped (negative) sum is 0.
i32 capped(i32 sum) { return sum < 0 ? 0 : (sum <= kEchoCap ? sum : kEchoCap); }

// The node of the first effect in effect with that stateInfo, or 0.
u64 effect_with(u64 first, u16 state) {
    for (u64 n = first; n; n = ld<u64>(n + kNext)) {
        if (ld<u32>(n + kState) & kNotInEffect) continue;
        const u64 p = ld<u64>(n + kParam);
        if (p && ld<u16>(p + kStateInfo) == state) return n;
    }
    return 0;
}

std::atomic<u64> g_discoveries{0}, g_gains{0}, g_kills{0}, g_penalties{0}, g_price_checks{0};

// ---- Item discovery (sub_1981830) ---------------------------------------
//
// `effects` is the character's holder, `arcane` its arcane. The effects'
// itemDropRate counts only while one of them is an item-discovery effect
// (and the holder's flags at +0x21 say there are such), and then every
// effect's counts, whatever its kind - a row-less node adds 0.

DECOMP_LEAF float item_discovery(u64 effects, float arcane) {
    g_discoveries.fetch_add(1, std::memory_order_relaxed);
    float bonus = 0.0f;
    if (effects && (ld<u8>(effects + 0x21) & 0x70)) {
        const u64 first = ld<u64>(effects + kFirst);
        if (effect_with(first, kStateDiscovery)) {
            for (u64 n = first; n; n = ld<u64>(n + kNext)) {
                if (ld<u32>(n + kState) & kNotInEffect) continue;
                const u64 p = ld<u64>(n + kParam);
                bonus = bonus + (p ? ld<float>(p + kItemDropRate) : 0.0f);
            }
        }
    }
    // The curve: 1 below 8, rising to 1.7 at 30 and 2.1 at 50, flat after
    // it. A NaN arcane fails every test and takes the top.
    float base;
    if (k(kDiscovery8) > arcane) {
        base = k(kDiscoveryOne);
    } else if (k(kDiscovery30) > arcane) {
        base = (arcane + k(kDiscoveryMinus8)) * k(kDiscoverySlope8) + k(kDiscoveryOne);
    } else if (k(kDiscovery50) > arcane) {
        base = (arcane + k(kDiscoveryMinus30)) * k(kDiscoverySlope30) + k(kDiscoveryBase30);
    } else {
        base = k(kDiscoveryTop);
    }
    const float r = bonus + base;
    return r > 0.0f ? r : 0.0f;  // vmaxss with 0: a NaN or -0 sum is +0
}

// ---- Echoes gained (PlayerIns::vf122) -------------------------------------
//
// `rates` (its low byte) applies the player's soulRate effects to `amount`
// first. Nothing moves unless the player has a record and is the local one
// (virtual +0x1b8, its low byte). The echoes stay in [0, 999,999,999] - a
// sum that wraps negative is 0 - and what they rose by also goes to the
// echoes ever gained (same rule, stored as 64 bits) and the HUD's counter
// +0x20c, `amount` (as applied) to +0x210. An effect of stateInfo 0x72 in
// effect gathers `amount` at +0xfc; without one the field is 0. The third
// argument is not read.

using LocalFn = GUEST_ABI u64 (*)(u64);

DECOMP_LEAF void add_echoes(u64 player, i32 amount, i32, u32 rates) {
    g_gains.fetch_add(1, std::memory_order_relaxed);
    float rate = 0.0f;
    if (rates & 0xff) {
        rate = k(kGainOne);
        for (u64 n = ld<u64>(ld<u64>(player + 0x1c8) + kFirst); n; n = ld<u64>(n + kNext)) {
            if (ld<u32>(n + kState) & kNotInEffect) continue;
            const u64 p = ld<u64>(n + kParam);
            rate = rate * (p ? ld<float>(p + kSoulRate) : k(kGainOne));
        }
    }
    if (!ld<u64>(player + 0x3c0)) return;
    const auto local = reinterpret_cast<LocalFn>(static_cast<std::uintptr_t>(ld<u64>(ld<u64>(player) + 0x1b8)));
    if (!static_cast<u8>(local(player))) return;
    const i32 add = (rates & 0xff) ? truncate(static_cast<float>(amount) * rate) : amount;
    const u64 rec = ld<u64>(player + 0x3c0);
    const i32 was = ld<i32>(rec + kEchoes);
    const i32 now = capped(wrap_add(was, add));
    st<i32>(rec + kEchoes, now);
    const i32 rose = wrap_sub(now, was);
    if (rose > 0) st<std::int64_t>(rec + kEchoesEver, capped(wrap_add(ld<i32>(rec + kEchoesEver), rose)));
    const u64 holder = ld<u64>(player + 0x1c8);
    i32 gathered = 0;
    if ((ld<u8>(holder + 0x22) & 0xe0) && effect_with(ld<u64>(holder + kFirst), kStateGathering))
        gathered = wrap_add(ld<i32>(rec + kEchoesGathered), add);
    st<i32>(rec + kEchoesGathered, gathered);
    if (!add) return;
    const u64 menu = ld<u64>(decomp_leaf_guest(kMenuMan));
    st<i32>(menu + 0x210, wrap_add(ld<i32>(menu + 0x210), add));
    st<i32>(menu + 0x20c, wrap_add(ld<i32>(menu + 0x20c), rose));
}

// ---- The echoes a kill gives (sub_1cfc860) --------------------------------
//
// `base` (the victim's echoes) times the victim's effects' haveSoulRate - by
// the clear count's scale where the effect is bGameClearBonus (index 0, a
// first playthrough, leaves it 1) - then by 0.5 when `halve`, by 1.2 for a
// cooperator, truncated and rounded up when what was cut is over 5e-6. A
// positive result goes to the player's echoes (virtual +0x3d0, with rates)
// and the kill on to sub_1c98290, whose result is this one's; otherwise the
// truncated amount is returned.

struct KillAmount {
    i32 truncated, rounded;
};

KillAmount kill_amount(u64 victim, u32 coop, u32 halve, float base) {
    float rate = k(kKillOne);
    if (victim) {
        const u64 first = ld<u64>(ld<u64>(victim + 0x1c8) + kFirst);
        if (first) {
            const u64 man = ld<u64>(decomp_leaf_guest(kGameDataMan));
            for (u64 n = first; n; n = ld<u64>(n + kNext)) {
                if (ld<u32>(n + kState) & kNotInEffect) continue;
                const u64 p = ld<u64>(n + kParam);
                float m = k(kKillOne);
                if (p) {
                    const float have = ld<float>(p + kHaveSoulRate);
                    if (!(ld<u8>(p + kClearBonusByte) & kClearBonusBit)) {
                        m = have;
                    } else {
                        i32 clears = ld<i32>(man + 0x68);
                        if (clears > 6) clears = 6;
                        const std::int64_t i = clears < 0 ? 0 : clears;
                        if (i) m = have * ld<float>(decomp_leaf_guest(kClearScale) + 4 * static_cast<u64>(i));
                    }
                }
                rate = rate * m;
            }
        }
    }
    const float half = (halve & 0xff) ? k(kKillHalf) : k(kKillOne);
    const float co = (coop & 0xff) ? k(kKillCoop) : k(kKillOne);
    float x = rate * base;
    x = half * x;
    x = co * x;
    const i32 t = truncate(x);
    const float cut = x - static_cast<float>(t);
    return {t, wrap_add(cut > k(kKillRoundUp) ? 1 : 0, t)};
}

using GainFn = GUEST_ABI void (*)(u64, i32, i32, u32);
using KillRecordFn = GUEST_ABI u64 (*)(u64, u64, u32, u32);

DECOMP_LEAF u64 kill_echoes(u64 player, u64 victim, u32 coop, u32 halve, float base) {
    g_kills.fetch_add(1, std::memory_order_relaxed);
    const KillAmount a = kill_amount(victim, coop, halve, base);
    if (a.rounded <= 0) return static_cast<u32>(a.truncated);
    reinterpret_cast<GainFn>(static_cast<std::uintptr_t>(ld<u64>(ld<u64>(player) + 0x3d0)))(player, a.rounded, 0, 1);
    return reinterpret_cast<KillRecordFn>(static_cast<std::uintptr_t>(decomp_leaf_guest(kKillRecord)))(player + 0x2a0, victim,
                                                                                                       coop & 0xff, 0);
}

// ---- The scripts' penalty (lua_cli_ExcutePenalty) --------------------------
//
// Only with the Lua event manager up and a record: the echoes lose
// `share` of themselves (truncated; a negative result is 0, the cap holds),
// insight loses `insight` (never below 0). What it returns is what the game's
// left in rax: the manager's slot, 0, or the record.

DECOMP_LEAF u64 ExcutePenalty(u64, i32 insight, float share) {
    g_penalties.fetch_add(1, std::memory_order_relaxed);
    const u64 slot = decomp_leaf_guest(kLuaEventMan);
    if (!ld<u64>(slot)) return slot;
    const u64 rec = ld<u64>(ld<u64>(decomp_leaf_guest(kGameDataMan)) + 0x8);
    if (!rec) return 0;
    const i32 echoes = ld<i32>(rec + kEchoes);
    i32 left = wrap_sub(echoes, truncate(static_cast<float>(echoes) * share));
    if (left < 0) left = 0;
    st<i32>(rec + kEchoes, left <= kEchoCap ? left : kEchoCap);
    i32 sane = wrap_sub(ld<i32>(rec + kInsight), insight);
    if (sane < 0) sane = 0;
    st<i32>(rec + kInsight, sane);
    return rec;
}

// ---- The level-up's price check (sub_1f2f5e0) ----------------------------
//
// One attribute from *from to *to on the level-up screen; the value it may
// take. Raising: the price of the planned level (the screen's +0xe94, as
// x = level + 81) is x^2 times CalcCorrectGraph row 200's curve, max(0, x -
// row+0x48) x row+0x44 + row+0x3c, plus row+0x40, rounded twice by +0.5 and
// truncating; it must be within the echoes less what the screen has already
// committed (+0xeb0), and 99 is the ceiling. Lowering: not below the value
// the screen opened with (*(ctx+0x10)).

struct ParamRef {
    i32 id;
    u32 pad;
    u64 row;
};
using ParamRowFn = GUEST_ABI void (*)(ParamRef*, i32);

DECOMP_LEAF u64 level_price_check(u64 ctx, const i32* from, const i32* to) {
    g_price_checks.fetch_add(1, std::memory_order_relaxed);
    const i32 was = *from, want = *to;
    if (want > was) {
        const u64 screen = ld<u64>(ctx + 0x8);
        const i32 level = ld<i32>(screen + 0xe94);
        ParamRef ref{-1, 0, 0};
        reinterpret_cast<ParamRowFn>(static_cast<std::uintptr_t>(decomp_leaf_guest(kParamRow)))(&ref, 0xc8);
        const float x = static_cast<float>(wrap_add(level, 81));
        float curve, price;
        if (ref.row) {
            float t = x - ld<float>(ref.row + 0x48);
            t = 0.0f > t ? 0.0f : t;  // vmaxss(0, t): a NaN t stays
            t = t * ld<float>(ref.row + 0x44);
            t = t + ld<float>(ref.row + 0x3c);
            curve = (x * x) * t;
            price = ld<float>(ref.row + 0x40);
        } else {
            float t = 0.0f > x ? 0.0f : x;
            t = t * 0.0f;
            t = t + 0.0f;
            curve = (x * x) * t;
            price = 0.0f;
        }
        if (want > 99) return static_cast<u32>(was);
        const i32 have = wrap_sub(ld<i32>(ld<u64>(ld<u64>(decomp_leaf_guest(kGameDataMan)) + 0x8) + kEchoes),
                                  ld<i32>(screen + 0xeb0));
        const float half = k(kPriceHalf);
        i32 c = truncate((curve + price) + half);
        c = truncate(static_cast<float>(c) + half);
        const i32 cost = static_cast<i32>(_mm_cvttss_si64(_mm_set_ss(static_cast<float>(c))));
        return static_cast<u32>(cost > have ? was : want);
    }
    if (want < was && ld<i32>(ld<u64>(ctx + 0x10)) > want) return static_cast<u32>(was);
    return static_cast<u32>(want);
}

// ---- Compare runs ---------------------------------------------------------
//
// Readers run both and compare. Writers run the game's, keep what it wrote,
// put the state back, run ours and compare - then the game's result stays.
// The kill reward cannot run twice (it pays and tells the session), so ours
// predicts the amount and the gain's compare watches what the game's pays.

void* g_game_discovery = nullptr;
void* g_game_gain = nullptr;
void* g_game_kill = nullptr;
void* g_game_penalty = nullptr;
void* g_game_price = nullptr;
DecompCompare g_cmp_discovery, g_cmp_gain, g_cmp_kill, g_cmp_penalty, g_cmp_price;

void count(DecompCompare& c, bool same, const char* fn, ull a, ull theirs, ull ours) {
    c.calls.fetch_add(1, std::memory_order_relaxed);
    if (same) return;
    if (c.differ.fetch_add(1, std::memory_order_relaxed) < 8)
        host_log("decomp: %s(0x%llx) differs: the game's 0x%llx, ours 0x%llx", fn, a, theirs, ours);
}

u32 bits(float f) {
    u32 b;
    std::memcpy(&b, &f, 4);
    return b;
}

using DiscoveryFn = GUEST_ABI float (*)(u64, float);
GUEST_ABI float compare_discovery(u64 effects, float arcane) {
    const float ours = item_discovery(effects, arcane);
    const float theirs = reinterpret_cast<DiscoveryFn>(g_game_discovery)(effects, arcane);
    count(g_cmp_discovery, bits(theirs) == bits(ours), "sub_1981830", bits(arcane), bits(theirs), bits(ours));
    return theirs;
}

// What a gain writes, to put back and compare.
struct GainState {
    i32 echoes, gathered, hud_210, hud_20c;
    std::int64_t ever;
};
GainState gain_state(u64 rec, u64 menu) {
    GainState s{};
    s.echoes = ld<i32>(rec + kEchoes);
    s.gathered = ld<i32>(rec + kEchoesGathered);
    s.ever = ld<std::int64_t>(rec + kEchoesEver);
    if (menu) {
        s.hud_210 = ld<i32>(menu + 0x210);
        s.hud_20c = ld<i32>(menu + 0x20c);
    }
    return s;
}
void set_gain_state(u64 rec, u64 menu, const GainState& s) {
    st<i32>(rec + kEchoes, s.echoes);
    st<i32>(rec + kEchoesGathered, s.gathered);
    st<std::int64_t>(rec + kEchoesEver, s.ever);
    if (menu) {
        st<i32>(menu + 0x210, s.hud_210);
        st<i32>(menu + 0x20c, s.hud_20c);
    }
}
bool same(const GainState& a, const GainState& b) {
    return a.echoes == b.echoes && a.gathered == b.gathered && a.ever == b.ever && a.hud_210 == b.hud_210 &&
           a.hud_20c == b.hud_20c;
}

// The last gain the game's code made, for the kill reward's compare.
struct SeenGain {
    bool seen;
    u64 player;
    i32 amount;
    u32 rates;
};
SeenGain g_seen_gain{};

GUEST_ABI void compare_gain(u64 player, i32 amount, i32 third, u32 rates) {
    g_seen_gain = {true, player, amount, rates};
    const u64 rec = ld<u64>(player + 0x3c0);
    const u64 menu = ld<u64>(decomp_leaf_guest(kMenuMan));
    if (!rec) {
        reinterpret_cast<GainFn>(g_game_gain)(player, amount, third, rates);
        return;
    }
    const GainState before = gain_state(rec, menu);
    reinterpret_cast<GainFn>(g_game_gain)(player, amount, third, rates);
    const GainState theirs = gain_state(rec, menu);
    set_gain_state(rec, menu, before);
    add_echoes(player, amount, third, rates);
    const GainState ours = gain_state(rec, menu);
    set_gain_state(rec, menu, theirs);
    count(g_cmp_gain, same(theirs, ours), "PlayerIns::vf122", static_cast<u32>(amount), static_cast<u32>(theirs.echoes),
          static_cast<u32>(ours.echoes));
}

using KillFn = GUEST_ABI u64 (*)(u64, u64, u32, u32, float);
GUEST_ABI u64 compare_kill(u64 player, u64 victim, u32 coop, u32 halve, float base) {
    const KillAmount ours = kill_amount(victim, coop, halve, base);
    g_seen_gain.seen = false;
    const u64 theirs = reinterpret_cast<KillFn>(g_game_kill)(player, victim, coop, halve, base);
    // The gain's own compare saw what the game paid - when it is placed.
    if (g_game_gain && decomp_comparing()) {
        const bool paid = g_seen_gain.seen && g_seen_gain.player == player;
        const bool ok = ours.rounded <= 0 ? !paid && theirs == static_cast<u32>(ours.truncated)
                                          : paid && g_seen_gain.amount == ours.rounded && (g_seen_gain.rates & 0xff) == 1;
        count(g_cmp_kill, ok, "sub_1cfc860", bits(base), paid ? static_cast<u32>(g_seen_gain.amount) : theirs,
              static_cast<u32>(ours.rounded));
    }
    return theirs;
}

using PenaltyFn = GUEST_ABI u64 (*)(u64, i32, float);
GUEST_ABI u64 compare_penalty(u64 a0, i32 insight, float share) {
    const u64 rec = ld<u64>(decomp_leaf_guest(kLuaEventMan)) ? ld<u64>(ld<u64>(decomp_leaf_guest(kGameDataMan)) + 0x8) : 0;
    if (!rec) return reinterpret_cast<PenaltyFn>(g_game_penalty)(a0, insight, share);
    const i32 echoes = ld<i32>(rec + kEchoes), sane = ld<i32>(rec + kInsight);
    const u64 theirs = reinterpret_cast<PenaltyFn>(g_game_penalty)(a0, insight, share);
    const i32 their_echoes = ld<i32>(rec + kEchoes), their_sane = ld<i32>(rec + kInsight);
    st<i32>(rec + kEchoes, echoes);
    st<i32>(rec + kInsight, sane);
    const u64 ours = ExcutePenalty(a0, insight, share);
    const bool ok = ours == theirs && ld<i32>(rec + kEchoes) == their_echoes && ld<i32>(rec + kInsight) == their_sane;
    st<i32>(rec + kEchoes, their_echoes);
    st<i32>(rec + kInsight, their_sane);
    count(g_cmp_penalty, ok, "lua_cli_ExcutePenalty", static_cast<u32>(echoes), static_cast<u32>(their_echoes),
          static_cast<u32>(ld<i32>(rec + kEchoes)));
    return theirs;
}

using PriceFn = GUEST_ABI u64 (*)(u64, const i32*, const i32*);
GUEST_ABI u64 compare_price(u64 ctx, const i32* from, const i32* to) {
    const u64 ours = level_price_check(ctx, from, to);
    const u64 theirs = reinterpret_cast<PriceFn>(g_game_price)(ctx, from, to);
    count(g_cmp_price, static_cast<u32>(theirs) == static_cast<u32>(ours), "sub_1f2f5e0", static_cast<u32>(*to),
          static_cast<u32>(theirs), static_cast<u32>(ours));
    return theirs;
}

void report() {
    host_log("decomp: player data: %llu discovery rolls, %llu echo gains, %llu kill rewards, %llu script penalties, %llu "
             "level-up price checks",
             static_cast<ull>(g_discoveries.load()), static_cast<ull>(g_gains.load()), static_cast<ull>(g_kills.load()),
             static_cast<ull>(g_penalties.load()), static_cast<ull>(g_price_checks.load()));
}

// The entries' first whole instructions, as the eboot has them.
constexpr u8 kDiscoveryEntry[] = {0xc5, 0xf0, 0x57, 0xc9, 0x48, 0x85, 0xff, 0x74, 0x72, 0xf6, 0x47, 0x21, 0x70, 0x74, 0x6c};
constexpr u8 kGainEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x83,
                             0xec, 0x10, 0x41, 0x89, 0xf6, 0x48, 0x89, 0xfb};
constexpr u8 kKillEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50,
                             0x41, 0x89, 0xd7, 0x49, 0x89, 0xf6, 0x48, 0x89, 0xfb};
constexpr u8 kPenaltyEntry[] = {0x48, 0x8d, 0x05, 0x81, 0x62, 0x20, 0x04, 0x48, 0x83, 0x38, 0x00, 0x74, 0x51};
constexpr u8 kPriceEntry[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                              0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x18};

}  // namespace

void decomp_player_data_add() {
    const char* area = "player data";
    DecompFunction discovery{"sub_1981830", area, 0x1981830, kDiscoveryEntry, sizeof(kDiscoveryEntry),
                             reinterpret_cast<void*>(&item_discovery), DecompKind::Leaf, &g_game_discovery,
                             reinterpret_cast<void*>(&compare_discovery), &g_cmp_discovery};
    discovery.report = &report;
    decomp_add(discovery);
    decomp_add({"PlayerIns::vf122", area, 0x1cfc600, kGainEntry, sizeof(kGainEntry), reinterpret_cast<void*>(&add_echoes),
                DecompKind::Leaf, &g_game_gain, reinterpret_cast<void*>(&compare_gain), &g_cmp_gain});
    decomp_add({"sub_1cfc860", area, 0x1cfc860, kKillEntry, sizeof(kKillEntry), reinterpret_cast<void*>(&kill_echoes),
                DecompKind::Leaf, &g_game_kill, reinterpret_cast<void*>(&compare_kill), &g_cmp_kill});
    decomp_add({"lua_cli_ExcutePenalty", area, 0x1734e40, kPenaltyEntry, sizeof(kPenaltyEntry),
                reinterpret_cast<void*>(&ExcutePenalty), DecompKind::Leaf, &g_game_penalty,
                reinterpret_cast<void*>(&compare_penalty), &g_cmp_penalty});
    decomp_add({"sub_1f2f5e0", area, 0x1f2f5e0, kPriceEntry, sizeof(kPriceEntry), reinterpret_cast<void*>(&level_price_check),
                DecompKind::Leaf, &g_game_price, reinterpret_cast<void*>(&compare_price), &g_cmp_price});
}
