#!/usr/bin/env python3
"""Generate a numpy matmul reference for check_sgemm --data.

Needs numpy:  pip install numpy

Output you're aiming for (raw float32, NO numpy header, so C++ can fread it):

    correctness/data/A.bin      M x K, row-major
    correctness/data/B.bin      K x N, row-major
    correctness/data/C_ref.bin  M x N, row-major = A @ B
    correctness/data/shape.txt  "M N K"

Usage:
    python3 correctness/reference.py                  # 512 512 512
    python3 correctness/reference.py 1024 1024 1024

YOUR TASK:
  1. argparse: optional positional M, N, K (default 512), --out, --seed.
  2. Make A and B with np.random.default_rng(seed), uniform in [-1, 1).
     Check the dtype you get by default — is it float32? It must be.
  3. C = A @ B. Decide: compute in float32 or float64? Write down what each
     choice would tell you when the GPU disagrees with it.
  4. Write the buffers. np.save adds a header; find the method that writes raw
     bytes. Is numpy's default memory order row-major? (Look up order='C'.)
  5. Write shape.txt.
  6. Sanity check: print |C| mean/max. If max is ~0, something is wrong.
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
