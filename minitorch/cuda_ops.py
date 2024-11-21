# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """Compiles a function to run on a CUDA device."""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """Compiles a function for CUDA execution."""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication on two tensors, adjusting for 2D and 3D shapes."""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_i = numba.cuda.blockIdx.x * THREADS_PER_BLOCK + numba.cuda.threadIdx.x
        if out_i < out.size:
            out_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            in_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            to_index(out_i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_position = index_to_position(in_index, in_strides)
            out_position = index_to_position(out_index, out_strides)
            out[out_position] = fn(in_storage[in_position])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_i = numba.cuda.blockIdx.x * THREADS_PER_BLOCK + numba.cuda.threadIdx.x
        if out_i < out.size:
            out_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            a_index = numba.cuda.local.array(MAX_DIMS, numba.int32)
            b_index = numba.cuda.local.array(MAX_DIMS, numba.int32)

            to_index(out_i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_position, b_position = (
                index_to_position(a_index, a_strides),
                index_to_position(b_index, b_strides),
            )
            out_position = index_to_position(out_index, out_strides)
            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    local_i = cuda.threadIdx.x

    cache[local_i] = 0
    if i < size:
        cache[local_i] = a[i]
    cuda.syncthreads()

    stride = BLOCK_DIM // 2
    while stride > 0:
        if local_i < stride:
            if i + stride < size:
                cache[local_i] += cache[local_i + stride]
        cuda.syncthreads()
        stride //= 2

    if local_i == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Sums elements of `a` in blocks, storing results in a new TensorData object."""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        if out_pos >= out_size:
            return

        # Convert output position to index
        to_index(out_pos, out_shape, out_index)
        out_position = index_to_position(out_index, out_strides)

        cache[pos] = reduce_value
        reduce_size = a_shape[reduce_dim]
        
        if pos < reduce_size:
            out_index[reduce_dim] = pos
            cache[pos] = a_storage[index_to_position(out_index, a_strides)]
        cuda.syncthreads()

        # Reduction in shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride and pos + stride < reduce_size:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        if pos == 0:
            out[out_position] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    cache_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    cache_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    if i < size and j < size:
        cache_a[i, j] = a[size * i + j]
        cache_b[i, j] = b[size * i + j]
        cuda.syncthreads()

        temp = 0.0
        for k in range(size):
            temp += cache_a[i, k] * cache_b[k, j]

        out[i * size + j] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication on two Tensors, returning a new TensorData."""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    BLOCK_DIM = 32
    shared_a = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    shared_b = numba.cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    y = numba.cuda.threadIdx.y
    x = numba.cuda.threadIdx.x
    block_x = numba.cuda.blockIdx.x * BLOCK_DIM
    block_y = numba.cuda.blockIdx.y * BLOCK_DIM
    z = numba.cuda.blockIdx.z

    temp = 0
    for block_index in range((a_shape[-1] + (BLOCK_DIM - 1)) // BLOCK_DIM):
        block_mid = block_index * BLOCK_DIM
        if (block_mid + x) < a_shape[-1] and (block_y + y) < a_shape[-2]:
            shared_a[y, x] = a_storage[
                z * a_batch_stride
                + (block_mid + x) * a_strides[-1]
                + (block_y + y) * a_strides[-2]
            ]
        else:
            shared_a[y, x] = 0
        if (block_x + x) < b_shape[-1] and (block_mid + y) < b_shape[-2]:
            shared_b[y, x] = b_storage[
                z * b_batch_stride
                + (block_x + x) * b_strides[-1]
                + (block_mid + y) * b_strides[-2]
            ]
        else:
            shared_b[y, x] = 0
        numba.cuda.syncthreads()

        for val in range(BLOCK_DIM):
            temp += shared_a[y, val] * shared_b[val, x]

    if (block_y + y) < out_shape[-2] and (block_x + x) < out_shape[-1]:
        out[
            z * out_strides[0]
            + (block_y + y) * out_strides[-2]
            + (block_x + x) * out_strides[-1]
        ] = temp


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
