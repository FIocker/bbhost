#!/usr/bin/env python3
"""Generate the FPS++ code sites engine/frame_rate.cpp applies beyond its own value tables, from Kyo's
"60 FPS++" and "90 FPS++" lists for Bloodborne 1.09 (shadPS4's patch list, user/patches/shadPS4/
Bloodborne.xml) and the eboot:

    tools/fps_sites.py Bloodborne.xml eboot-109-decrypted.bin

writes src/engine/sixty_fps_sites.inc and src/engine/ninety_fps_sites.inc. Addresses in the lists are
Binary Ninja's (the ELF's VA + 0x400000), the same as frame_rate.cpp's.

Each list's lines are applied in order to a copy of the image (later lines overwrite earlier ones, as a
patch loader does), then every byte the list changes is kept except:
  - bytes engine/frame_rate.cpp's value tables write (kConstants, kCounts, kBytes): those values are
    computed there for the rate the game runs at;
  - the frame-time manager, 0x2434770..0x2435480 - ours replaces that function whole
    (its 60 mode is case 2 with no history, which is what those edits make of it);
  - EXCLUDE below, each with the reason.
Contiguous kept bytes become one site: its address, the bytes the eboot holds there, the bytes the list
leaves. frame_rate.cpp checks every site's old bytes before writing any.

The 90 file holds only what "90 FPS++" does beyond "60 FPS++": a run of the bytes either list changes is
kept when any byte in it differs from what the 60 sites leave. At 90 frame_rate.cpp writes the 60 sites,
then these over them, so the old bytes recorded are the eboot's. The tool checks that each cloth cave reads
the scalars the 90 list stores at 0x4d3bd00 and that the damping exponent's new load reads 2/3.
"""
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

FRAME_MANAGER = (0x2434770, 0x2435480)
EXCLUDE = [
    # sub_2435630: a local's zeroing turned into a store of mode 2 into the flipper; ours picks the mode.
    (0x243565e, 0x2435666, 'frame mode'),
    # The frame modes' name table (debug display); absolute pointers valid only at the default load
    # address, and relocated at load, so not what the file holds.
    (0x575a920, 0x575a928, 'mode names'), (0x575a930, 0x575a938, 'mode names'),
    # The engine's timed condition wait returns at once instead of waiting: every job worker spins.
    # An emulator's latency workaround; the host's waits wake on time.
    (0x2483ec1, 0x2483ec6, 'timed wait'),
    # The sound thread's sleep, capped through a cave written over the tail of a live function
    # (sub_4632d0); frame_rate.cpp sets that thread's update time instead (MainThreadUpdateTime).
    (0x463300, 0x463312, 'sleep cave'), (0x2c0137d, 0x2c01382, 'sleep cave'),
    # The later list's version of that cap: the stub written into the loop itself (0x2c012cc).
    (0x2c012cc, 0x2c012de, 'sleep cave'),
    # The same loop's three profiling clock reads skipped: not needed with a host gettimeofday.
    (0x2c0118a, 0x2c0118c, 'profiling'), (0x2c011eb, 0x2c011ed, 'profiling'), (0x2c012ca, 0x2c012cc, 'profiling'),
]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CLOTH_SCALARS = 0x4d3bd00  # 4 x constraint stiffness, then 4 x the integrator's acceleration scale


def load_eboot(path):
    eb = open(path, 'rb').read()
    phoff, phnum = struct.unpack_from('<Q', eb, 0x20)[0], struct.unpack_from('<H', eb, 0x38)[0]
    segs = []
    for i in range(phnum):
        t, fl, off, va, pa, fs, ms, al = struct.unpack_from('<IIQQQQQQ', eb, phoff + i * 56)
        if t == 1:
            segs.append((va + 0x400000, off, fs))

    def orig(a):
        for va, off, fs in segs:
            if va <= a < va + fs:
                return eb[off + a - va]
        raise SystemExit('0x%x is not in the image' % a)
    return orig


def list_image(root, name):
    md = next((m for m in root.iter('Metadata') if m.attrib.get('Name') == name and m.attrib.get('AppVer') == '01.09'),
              None)
    if md is None:
        raise SystemExit('no "%s" for 01.09 in the list' % name)
    image = {}
    for line in md.find('PatchList'):
        a, ty, v = int(line.attrib['Address'], 16), line.attrib['Type'], line.attrib['Value']
        data = bytes.fromhex(v) if ty == 'bytes' else struct.pack('<I', int(v, 16)) if ty == 'bytes32' else None
        if data is None:
            raise SystemExit('line type %s' % ty)
        for k, b in enumerate(data):
            image[a + k] = b
    return image


def table_bytes():
    """Every byte frame_rate.cpp's value tables write."""
    src = open(os.path.join(ROOT, 'src/engine/frame_rate.cpp')).read()
    ours = set()
    # kConstants: {0xADDR, 0xWAS, 0xNOW[, Kind::...]}
    for m in re.finditer(r'\{0x([0-9a-f]+), 0x[0-9a-f]+, 0x[0-9a-f]+[,}]', src):
        ours.update(range(int(m.group(1), 16), int(m.group(1), 16) + 4))
    # kCounts and kBytes: {0xADDR, N, ...}
    for m in re.finditer(r'\{0x([0-9a-f]+), (\d+), ', src):
        ours.update(range(int(m.group(1), 16), int(m.group(1), 16) + int(m.group(2))))
    return ours


def changed_runs(image, orig, ours, also=()):
    """Runs of the bytes the list changes (and `also`), less the ones left out."""
    kept = sorted(a for a in set(a for a, b in image.items() if b != orig(a)) | set(also)
                  if a not in ours and not FRAME_MANAGER[0] <= a < FRAME_MANAGER[1]
                  and not any(lo <= a < hi for lo, hi, _ in EXCLUDE))
    # One site per run of changed bytes, joined across up to three unchanged bytes of the list's own
    # writes so a rewritten instruction stays whole.
    sites = []
    for a in kept:
        if sites and a - (sites[-1][0] + sites[-1][1]) <= 3 and all(x in image for x in range(sites[-1][0] + sites[-1][1], a)):
            sites[-1][1] = a - sites[-1][0] + 1
        else:
            sites.append([a, 1])
    return sites


def write_inc(path, header, sites, image, orig):
    total = 0
    with open(path, 'w', newline='\n') as f:
        for line in header:
            f.write('// ' + line + '\n')
        for a, n in sites:
            old = bytes(orig(a + k) for k in range(n))
            new = bytes(image.get(a + k, orig(a + k)) for k in range(n))
            total += n
            f.write('    {0x%08x, "%s", "%s"},\n' % (a, old.hex(), new.hex()))
        f.write('// %d sites, %d bytes\n' % (len(sites), total))
    print('%s: %d sites, %d bytes' % (path, len(sites), total))


def rip_target(image, orig, at, size, disp_at):
    """The address a RIP-relative operand reads: the instruction at `at`, `size` bytes, its rel32 at disp_at."""
    rel = struct.unpack('<i', bytes(image.get(disp_at + k, orig(disp_at + k)) for k in range(4)))[0]
    return at + size + rel


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root = ET.parse(sys.argv[1]).getroot()
    orig = load_eboot(sys.argv[2])
    ours = table_bytes()
    sixty = list_image(root, '60 FPS++')
    ninety = list_image(root, '90 FPS++')

    sites60 = changed_runs(sixty, orig, ours)
    write_inc(os.path.join(ROOT, 'src/engine/sixty_fps_sites.inc'),
              ['Generated by tools/fps_sites.py from Kyo\'s "60 FPS++" list for 1.09 (shadPS4\'s patch list,',
               'user/patches/shadPS4/Bloodborne.xml) and the 1.09 eboot: address (Binary Ninja), old bytes, new bytes.'],
              sites60, sixty, orig)

    # What the 60 sites leave, byte for byte. A 90 site is a run of the bytes
    # the 90 list or a 60 site changes, kept when any byte of it is not what
    # the 60 sites leave there - also a byte the 60 list changes and the 90
    # list leaves as the eboot has it (0xb6fbd4's first: 0x70 at 90, 0x04 at
    # 60), which the 90 site writes back.
    after60 = {}
    for a, n in sites60:
        for k in range(n):
            after60[a + k] = sixty.get(a + k, orig(a + k))
    sites90 = [s for s in changed_runs(ninety, orig, ours, also=[a for a, b in after60.items() if b != orig(a)])
               if any(ninety.get(s[0] + k, orig(s[0] + k)) != after60.get(s[0] + k, orig(s[0] + k)) for k in range(s[1]))]

    # The cloth package: every cave's vmulps reads the stiffness scalars, the integrator's reads the
    # acceleration scale beside them, and the damping exponent's load reads 2/3.
    def f32_at(a):
        return struct.unpack('<f', bytes(ninety.get(a + k, orig(a + k)) for k in range(4)))[0]
    caves = [a for a, n in sites90 if n == 25 and ninety.get(a) == 0xeb and ninety.get(a + 2) == 0xc5]
    for a in caves:
        if ninety.get(a + 3) == 0xf8:  # vmulps xmm0, xmm0, [rip+d]: c5 f8 59 05 d32
            t = rip_target(ninety, orig, a + 2, 8, a + 6)
            if t != CLOTH_SCALARS:
                raise SystemExit('cave 0x%x reads 0x%x, not the stiffness scalars' % (a, t))
        else:  # vmulss xmm0, xmm0, xmm0; vmulss xmm0, xmm0, [rip+d]
            t = rip_target(ninety, orig, a + 6, 8, a + 10)
            if t != CLOTH_SCALARS + 0x10:
                raise SystemExit('cave 0x%x reads 0x%x, not the acceleration scale' % (a, t))
    if len(caves) != 7:
        raise SystemExit('%d cloth caves in the 90 list, not 7' % len(caves))
    damping = 0xb6fbd0
    t = rip_target(ninety, orig, damping, 8, damping + 4)
    if abs(f32_at(t) - 2.0 / 3.0) > 1e-6:
        raise SystemExit('the damping exponent at 90 reads %g at 0x%x' % (f32_at(t), t))
    for k in range(4):
        if abs(f32_at(CLOTH_SCALARS + 4 * k) - 2.0 / 3.0) > 1e-6 or abs(f32_at(CLOTH_SCALARS + 0x10 + 4 * k) - 1.5) > 1e-6:
            raise SystemExit('the cloth scalars at 90 are not 2/3 and 1.5')
    write_inc(os.path.join(ROOT, 'src/engine/ninety_fps_sites.inc'),
              ['Generated by tools/fps_sites.py from Kyo\'s "90 FPS++" list for 1.09 (shadPS4\'s patch list,',
               'user/patches/shadPS4/Bloodborne.xml) and the 1.09 eboot: what it does beyond "60 FPS++", written over',
               'the 60 sites at 90 and uncapped. Address (Binary Ninja), old bytes (the eboot\'s), new bytes.'],
              sites90, ninety, orig)


if __name__ == '__main__':
    main()
