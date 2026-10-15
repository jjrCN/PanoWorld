"""Patch gsplat's JIT loader so it uses MSVC-compatible compiler flags on Windows.

gsplat 1.5.3 hard-codes GCC flags ("-O3", "-Wno-attributes") for the host
compiler, which cl.exe rejects (error D8021). Run once after installing gsplat:

    .venv\Scripts\python.exe scripts\patch_gsplat_windows.py
"""
import pathlib
import gsplat

p = pathlib.Path(gsplat.__file__).parent / "cuda" / "_backend.py"
s = p.read_text(encoding="utf-8")
old = '        extra_cflags = [opt_level, "-Wno-attributes"]\n'
new = (
    '        if os.name == "nt":\n'
    '            # MSVC does not understand GCC flags.\n'
    '            extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]\n'
    '        else:\n'
    '            extra_cflags = [opt_level, "-Wno-attributes"]\n'
)
if new in s:
    print("already patched:", p)
elif old in s:
    p.write_text(s.replace(old, new), encoding="utf-8", newline="\n")
    print("patched:", p)
else:
    raise SystemExit(f"unexpected gsplat/_backend.py content, not patched: {p}")
