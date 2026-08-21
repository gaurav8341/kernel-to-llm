# common/

Shared infrastructure used across `projects/pXX/` — never the learning content itself. See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the reasoning.

- `include/` — generic helpers (timing wrappers, correctness-check utilities) usable from any project
- `ptx_sass/` — PTX/SASS dump scripts, used by the projects with an optional "Going Lower" section (P01, P04, P06)

Nothing here should contain a kernel — kernels live in their own `projects/pXX/src/`.
