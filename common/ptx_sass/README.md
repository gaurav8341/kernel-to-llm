# common/ptx_sass/

Scripts for dumping and reading PTX/SASS from a compiled kernel, used by the optional "Going Lower" sections in P01, P04, and P06 (per `Objective.md`).

- `dump_ptx.sh <kernel.cu> [out_dir]` — `nvcc --ptx`, then prints a histogram of the
  opcodes worth watching (`ld.global`, `ld.global.nc`, `ld.shared`, `fma.rn.f32`,
  `bar.sync`, `st.local`/`ld.local`) and warns on local-memory traffic (register spill).
- `dump_sass.sh <kernel.cu> [out_dir]` — `nvcc -cubin` + `cuobjdump --dump-sass`, then the
  equivalent histogram (`LDG`/`STG`, `LDS`/`STS`, `FFMA`, `BAR.SYNC`, `LDL`/`STL`) plus the
  `-Xptxas -v` register and spill counts.

Both default to `reports/ptx` / `reports/sass` under the caller's cwd, take `-arch=sm_86`
via the `ARCH` env var, and pass trailing arguments through to `nvcc` (e.g. `-DTILE_DIM=16`).
Wired into P01 as `make ptx` / `make sass`.
