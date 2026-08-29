#!/usr/bin/env python3
"""
patch_dlssnr.py - unlock the DLSS-NR NGX snippet's architecture gates on all four
backends (D3D11 / D3D12 / Vulkan / CUDA).

Signature-driven, so it survives offset changes between snippet builds.
It patches CODE GATES ONLY. It cannot create SASS for an architecture whose
cubins are not already present in the file - see --check.

  python3 patch_dlssnr.py --check  in.dll
  python3 patch_dlssnr.py in.dll -o out.dll [--arch 0x190]
"""
import argparse, re, struct, sys, shutil

def sections(buf):
    pe = struct.unpack_from("<I", buf, 0x3c)[0]
    assert buf[pe:pe+4] == b"PE\0\0", "not a PE"
    nsec  = struct.unpack_from("<H", buf, pe+6)[0]
    optsz = struct.unpack_from("<H", buf, pe+20)[0]
    base  = struct.unpack_from("<Q", buf, pe+24+24)[0]
    tbl   = pe+24+optsz
    out=[]
    for i in range(nsec):
        e=tbl+i*40
        out.append(dict(name=buf[e:e+8].rstrip(b"\0").decode(),
                        vsize=struct.unpack_from("<I",buf,e+8)[0],
                        vaddr=struct.unpack_from("<I",buf,e+12)[0],
                        rsize=struct.unpack_from("<I",buf,e+16)[0],
                        raw  =struct.unpack_from("<I",buf,e+20)[0]))
    return base, out

def mk_conv(base, secs):
    def va2off(va):
        rva = va - base
        for s in secs:
            if s["vaddr"] <= rva < s["vaddr"] + max(s["vsize"], s["rsize"]):
                return s["raw"] + (rva - s["vaddr"])
    def off2va(off):
        for s in secs:
            if s["raw"] <= off < s["raw"] + s["rsize"]:
                return base + s["vaddr"] + (off - s["raw"])
    return va2off, off2va

# ---------- gate 1: MinHWArchitecture reported by *_GetFeatureRequirements ----------
# c7 44 24 XX 12 00 00 00      mov [rsp+XX], 0x12     (NVSDK_NGX_Feature_DLSSNR)
# ...<=8 bytes...
# c7 44 24 YY ?? ?? 00 00      mov [rsp+YY], minArch   (YY == XX+4)
def find_minarch(buf, text):
    lo, hi = text["raw"], text["raw"]+text["rsize"]
    hits=[]
    for m in re.finditer(rb"\xc7\x44\x24(.)\x12\x00\x00\x00", buf[lo:hi]):
        slot = m.group(1)[0]
        win  = buf[lo+m.end(): lo+m.end()+12]
        m2 = re.search(rb"\xc7\x44\x24(.)(..)\x00\x00", win)
        if m2 and m2.group(1)[0] == slot+4:
            hits.append((lo+m.end()+m2.start()+4, struct.unpack("<H", m2.group(2))[0]))
    return hits

# ---------- gate 2: architecture switch inside *_CreateFeature ----------
def find_switch(buf, base, secs, va2off, off2va):
    text = next(s for s in secs if s["name"].startswith(".text"))
    lo, hi = text["raw"], text["raw"]+text["rsize"]
    sva=None
    for s in secs:
        i = buf[s["raw"]:s["raw"]+s["rsize"]].find(b"Unsupported GPU architecture")
        if i >= 0:
            while i > 0 and buf[s["raw"]+i-1] != 0: i -= 1
            sva = base + s["vaddr"] + i; break
    if sva is None: return None
    logsite=None
    for i in range(lo, hi-7):
        if buf[i] in (0x48,0x4c) and buf[i+1]==0x8d and (buf[i+2]&0xC7)==0x05:
            if off2va(i)+7+struct.unpack_from("<i",buf,i+3)[0] == sva:
                logsite=i; break
    if logsite is None: return None
    win_lo = max(lo, logsite-0x200)
    blk = buf[win_lo:logsite]
    m_lea = re.search(rb"\x8d\x88(....)", blk)                      # lea ecx,[rax-imm]
    m_idx = re.search(rb"\x41\x0f\xb6\x8c\x08(....)", blk)          # movzx ecx,[r8+rcx+d]
    m_tbl = re.search(rb"\x41\x8b\x94\x88(....)", blk)              # mov edx,[r8+rcx*4+d]
    m_r8  = re.search(rb"\x4c\x8d\x05(....)", blk)                  # lea r8,[rip+d]
    m_cnt = re.search(rb"\x83\xf9(.)\x77(.)", blk)                  # cmp ecx,count / ja default
    if not all((m_lea,m_idx,m_tbl,m_r8,m_cnt)): return None
    dflt = off2va(win_lo+m_cnt.end())+m_cnt.group(2)[0]
    archbase = -struct.unpack("<i", m_lea.group(1))[0]
    r8off    = win_lo+m_r8.start()
    r8val    = off2va(r8off)+7+struct.unpack("<i", m_r8.group(1))[0]
    idx_va   = r8val + struct.unpack("<I", m_idx.group(1))[0]
    tbl_va   = r8val + struct.unpack("<I", m_tbl.group(1))[0]
    return dict(archbase=archbase, count=m_cnt.group(1)[0]+1, dflt=dflt,
                idx_off=va2off(idx_va), tbl_off=va2off(tbl_va),
                r8val=r8val, fail_va=off2va(logsite))

def follow(buf, va, va2off, depth=3):
    """A case stub may have been rewritten to jump elsewhere (this is how the
    circulating community build unlocks Ada - it rewrites the stub, not the
    table). Follow the first unconditional jmp in the stub so the reported
    target is where control actually ends up."""
    for _ in range(depth):
        o = va2off(va)
        if o is None: return va
        stub = buf[o:o+16]
        nxt = None
        for k in range(len(stub)-1):
            if stub[k] == 0xEB:                       # jmp rel8
                nxt = va + k + 2 + struct.unpack_from("<b", stub, k+1)[0]; break
            if stub[k] == 0xE9 and k+5 <= len(stub):  # jmp rel32
                nxt = va + k + 5 + struct.unpack_from("<i", stub, k+1)[0]; break
            if stub[k] in (0xC3, 0xCC): break
        if nxt is None or nxt == va: return va
        va = nxt
    return va

def switch_map(buf, sw, off2va, va2off=None):
    rows=[]
    for i in range(sw["count"]):
        idx = buf[sw["idx_off"]+i]
        tgt = sw["r8val"] + struct.unpack_from("<I", buf, sw["tbl_off"]+idx*4)[0]
        eff = follow(buf, tgt, va2off) if va2off else tgt
        rows.append((sw["archbase"]+i, idx, tgt, eff))
    return rows

ARCH = {0x140:"Volta", 0x150:"GV110", 0x160:"Turing (RTX 20)", 0x170:"Ampere (RTX 30)",
        0x180:"Hopper", 0x190:"Ada (RTX 40)", 0x1a0:"Blackwell DC", 0x1b0:"Blackwell2 (RTX 50)"}

def cubin_archs(buf):
    try: import zstandard as zstd
    except ImportError: return None
    import io
    d=zstd.ZstdDecompressor(); found=set()
    for m in re.finditer(rb"\x28\xb5\x2f\xfd", buf):
        try: out=d.stream_reader(io.BytesIO(buf[m.start():m.start()+0x4000000])).read(96*1024*1024)
        except Exception: continue
        found |= {x.decode() for x in re.findall(rb"sm_\d+", out)}
    found |= {x.decode().split()[-1] for x in re.findall(rb"arch sm_\d+", buf)}
    return found

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dll"); ap.add_argument("-o","--out")
    ap.add_argument("--arch", default="0x190")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--no-gate2", action="store_true",
                    help="patch only the four GetFeatureRequirements constants; "
                         "leave the CreateFeature switch alone (minimal diff)")
    A=ap.parse_args()
    want=int(A.arch,0)
    buf=bytearray(open(A.dll,"rb").read())
    base,secs=sections(buf); va2off,off2va=mk_conv(base,secs)
    text=next(s for s in secs if s["name"].startswith(".text"))

    print(f"[*] {A.dll}  imagebase 0x{base:x}")
    ca=cubin_archs(buf)
    if ca is not None:
        print(f"[*] cubin SASS targets present: {', '.join(sorted(ca)) or 'none'}")
        need={0x190:"sm_89",0x1b0:"sm_120",0x170:"sm_86",0x160:"sm_75"}.get(want)
        if need and need not in ca:
            print(f"[!] WARNING: no {need} cubins in this file - {ARCH.get(want,hex(want))} "
                  f"will pass the gates and then fail at kernel load.")

    print("\n[*] gate 1 - MinHWArchitecture reported by *_GetFeatureRequirements")
    ma=find_minarch(buf,text)
    if len(ma)!=4: print(f"[!] expected 4 sites, found {len(ma)}")
    for off,val in ma:
        print(f"      file 0x{off:06x}  va 0x{off2va(off):x}  = 0x{val:x} ({ARCH.get(val,'?')})")

    print("\n[*] gate 2 - architecture switch in *_CreateFeature")
    sw=find_switch(buf,base,secs,va2off,off2va)
    if not sw: print("[!] not found"); return 1
    print(f"      table base 0x{sw['r8val']:x}  idx@file 0x{sw['idx_off']:x}  "
          f"targets@file 0x{sw['tbl_off']:x}  entries {sw['count']}")
    rows=switch_map(buf,sw,off2va,va2off)
    allow_idx=None
    for arch,idx,tgt,eff in rows:
        if tgt>sw["fail_va"]: allow_idx=idx
    for arch,idx,tgt,eff in rows:
        if arch%0x10: continue
        if eff>sw["fail_va"]:      st="ALLOW" + (" (stub rewritten)" if eff!=tgt else "")
        elif tgt==sw["dflt"]:      st="default->allow"
        else:                      st="block"
        via = "" if eff==tgt else f" =>0x{eff:x}"
        print(f"      0x{arch:03x} {ARCH.get(arch,''):<22} idx={idx} -> 0x{tgt:x}{via}  {st}")
    if allow_idx is None: print("[!] no allow target found"); return 1

    if A.check: return 0
    if not A.out: print("\n[!] need -o OUT to write"); return 1

    print(f"\n[*] patching: minimum architecture -> 0x{want:x} ({ARCH.get(want,'?')})")
    n=0
    for off,val in ma:
        if val!=want:
            struct.pack_into("<H",buf,off,want); n+=1
            print(f"      file 0x{off:06x}: 0x{val:x} -> 0x{want:x}")
    if A.no_gate2:
        print("      (gate 2 left untouched: --no-gate2)")
    for arch,idx,tgt,eff in rows:
        if A.no_gate2: break
        if arch%0x10: continue          # gap ids - no such GPU, leave them alone
        if eff>sw["fail_va"]: continue  # already reaches allow (table or rewritten stub)
        if arch>=want and idx!=allow_idx and tgt!=sw["dflt"]:
            buf[sw["idx_off"]+(arch-sw["archbase"])]=allow_idx; n+=1
            print(f"      file 0x{sw['idx_off']+(arch-sw['archbase']):06x}: "
                  f"switch 0x{arch:03x} idx {idx} -> {allow_idx} (ALLOW)")
    if n==0: print("      nothing to do - already patched")
    open(A.out,"wb").write(bytes(buf))
    print(f"\n[+] wrote {A.out}  ({n} change(s))")
    print("[!] Authenticode signature is now invalid; only usable via a loader that")
    print("    LoadLibrary's the snippet directly (nvngx.dll will reject it).")
    return 0

sys.exit(main())
