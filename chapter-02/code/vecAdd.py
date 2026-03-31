from pathlib import Path
from time import time

import torch
from torch.utils.cpp_extension import load_inline


def vector_multipication_loop(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.device == B.device and A.device.type == "cuda"
    assert A.dtype == B.dtype == torch.float32
    assert A.size() == B.size()

    C = torch.empty_like(A)
    n = A.size(0)
    for i in range(n):
        C[i] = A[i] + B[i]
    return C


def compile_extension():
    cuda_source = (Path(__file__).parent / "vecAddTorchTensor.cu").read_text()
    cpp_source = (
        "torch::Tensor vector_add(torch::Tensor A_h, torch::Tensor B_h);"
    )

    return load_inline(
        name="extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["vector_add"],
        with_cuda=True,
        # extra_cuda_cflags=["-O2"]
    )


def main():
    ext = compile_extension()

    DEVICE = "cuda"
    DTYPE = torch.float32
    NUM_ELEMENTS = 500_000

    A = torch.tensor([i for i in range(NUM_ELEMENTS)]).to(DEVICE, DTYPE, non_blocking=True)
    B = torch.tensor([i for i in range(NUM_ELEMENTS)]).to(DEVICE, DTYPE, non_blocking=True)

    start = time()
    C_cuda = ext.vector_add(A, B)
    stop = time()
    print(f"Cuda custom kernel multiply: {stop - start:.6f}s")

    start = time()
    C_loop = vector_multipication_loop(A, B)
    stop = time()
    print(f"Python loop: {stop - start:.6f}s")

    start = time()
    C_torch = A + B
    stop = time()
    print(f"PyTorch *: {stop - start:.6f}s")

    print("Size:", C_cuda.size())
    print("C_cuda:", C_cuda[:10])
    print("C_loop:", C_loop[:10])
    print("C_torch:", C_torch[:10])

    torch.testing.assert_close(C_cuda, C_torch)


if __name__ == "__main__":
    main()
