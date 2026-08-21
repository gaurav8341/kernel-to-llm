# common/ptx_sass/

Scripts for dumping and reading PTX/SASS from a compiled kernel, used by the optional "Going Lower" sections in P01, P04, and P06 (per `Objective.md`).

Expected scripts (added when first needed, e.g. in P01):
- `dump_ptx.sh` — `nvcc --ptx <kernel>.cu -o <kernel>.ptx`
- `dump_sass.sh` — `nvcc -cubin <kernel>.cu -o <kernel>.cubin && cuobjdump --sass <kernel>.cubin`
