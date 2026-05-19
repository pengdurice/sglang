// Fused across-heads RMSNorm + interleave (GPT-J) RoPE for diffusion DiTs.
//
// Combines the across-heads QK-Norm reduction from
//   python/sglang/jit_kernel/csrc/elementwise/qknorm_across_heads.cuh
// with the cos_sin_cache-style RoPE from
//   python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh
// into a single in-place pass for the `qk_norm == "rms_norm_across_heads"`
// branch used by Wan / Sana / Helios (and other across-heads DiTs).
//
// Layout:
//   q / k       : [num_tokens, dim]    where dim = num_heads * head_dim
//   q_weight    : [dim]                (one weight per element of `dim`)
//   k_weight    : [dim]
//   cos_sin_cache : [max_pos, rope_dim] float32, layout [cos | sin] along last
//   positions   : [num_tokens]         int32 or int64
//
// One CTA processes one (token, qk) pair: gridDim = (num_tokens, 2). All
// threads in the CTA cooperate on the across-heads sum-of-squares reduction;
// after the rsqrt is broadcast through shared memory, each thread applies
// the per-head RoPE on the slice of `dim` it owns (only the first `rope_dim`
// elements of each head are rotated; the rest pass through normed).
//
// Constraints (enforced at JIT-load time by the wrapper):
//   * dtype in {fp16, bf16}
//   * dim <= 8192 (16-byte vec) or 12288 (32-byte vec)
//   * head_dim and rope_dim are multiples of (vec_bytes / dtype_bytes)
//   * RoPE style is interleave / GPT-J (is_neox=False). NeoX is out of
//     scope for this kernel; the existing per-head fused path supports it
//     and across-heads diffusion configs (Wan/Sana/Helios) all use
//     interleave.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <cooperative_groups/reduce.h>
#include <tvm/ffi/container/tensor.h>

#include <cooperative_groups.h>
#include <type_traits>

namespace {

template <typename T, int VEC_SIZE_IN_BYTE>
struct VecTypeTrait;

template <>
struct VecTypeTrait<bf16_t, 16> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct VecTypeTrait<fp16_t, 16> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 4>;
};

template <>
struct VecTypeTrait<bf16_t, 32> {
  using packed_t = packed_t<bf16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <>
struct VecTypeTrait<fp16_t, 32> {
  using packed_t = packed_t<fp16_t>;
  using vec_t = device::AlignedVector<packed_t, 8>;
};

template <typename T, int VEC_SIZE_IN_BYTE, int kHeadDim, int kRopeDim, typename IdType>
__global__ void qknorm_across_heads_rope_kernel(
    T* __restrict__ q,
    T* __restrict__ k,
    const T* __restrict__ q_weight,
    const T* __restrict__ k_weight,
    const float* __restrict__ cos_sin_cache,
    const IdType* __restrict__ positions,
    int vec_hidden_size,
    float eps) {
  constexpr int inner_loop = VEC_SIZE_IN_BYTE == 16 ? 4 : 8;
  constexpr int kElemsPerVec = inner_loop * 2;
  static_assert(kHeadDim % kElemsPerVec == 0, "head_dim must align with vec width");
  static_assert(kRopeDim % kElemsPerVec == 0, "rope_dim must align with vec width");
  static_assert(kRopeDim <= kHeadDim, "rope_dim must not exceed head_dim");
  static_assert(kRopeDim % 2 == 0, "rope_dim must be even (paired interleave RoPE)");

  __shared__ float shared_memory[32];

  using vec_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::vec_t;
  using packed_t = typename VecTypeTrait<T, VEC_SIZE_IN_BYTE>::packed_t;
  vec_t v_data;
  vec_t v_weight;
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  const int warp_count = (blockDim.x + 31) >> 5;
  const float inv_hidden_size = 1.0f / static_cast<float>(vec_hidden_size * kElemsPerVec);
  const bool is_q = blockIdx.y == 0;

  const auto token_id = blockIdx.x;
  float2 acc_square = make_float2(0.0f, 0.0f);
  vec_t* data = reinterpret_cast<vec_t*>(is_q ? q : k) + token_id * vec_hidden_size;
  const vec_t* weight = reinterpret_cast<const vec_t*>(is_q ? q_weight : k_weight);

  // ---------------------------------------------------------------------
  // Phase 1: load q (or k), accumulate sum-of-squares across the full
  // hidden dimension (this is the "across heads" reduction).
  // ---------------------------------------------------------------------
  if (threadIdx.x < vec_hidden_size) {
    v_data = data[threadIdx.x];
    v_weight = weight[threadIdx.x];
    for (int i = 0; i < inner_loop; i++) {
      float2 val = device::cast<fp32x2_t, packed_t>(v_data[i]);
      acc_square.x += val.x * val.x;
      acc_square.y += val.y * val.y;
    }
  }

  auto cg_warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  float* buffer = shared_memory;
  float warp_sum = cooperative_groups::reduce(cg_warp, acc_square.x + acc_square.y, cooperative_groups::plus<float>());
  if (lane_id == 0) {
    buffer[warp_id] = warp_sum;
  }

  __syncthreads();
  if (threadIdx.x < 32) {
    float cta_sum = cooperative_groups::reduce(
        cg_warp, (threadIdx.x < warp_count) ? buffer[threadIdx.x] : 0.0f, cooperative_groups::plus<float>());
    if (threadIdx.x == 0) {
      buffer[0] = rsqrtf(eps + cta_sum * inv_hidden_size);
    }
  }
  __syncthreads();

  // ---------------------------------------------------------------------
  // Phase 2: scale by weight + rsqrt; then apply per-head interleave RoPE
  // on the first rope_dim elements of each head, in registers, before
  // writing the vec_t back to global.
  // ---------------------------------------------------------------------
  if (threadIdx.x < vec_hidden_size) {
    const float rsqrt_val = buffer[0];

    float elems[kElemsPerVec];
#pragma unroll
    for (int i = 0; i < inner_loop; i++) {
      float2 val = device::cast<fp32x2_t, packed_t>(v_data[i]);
      float2 wval = device::cast<fp32x2_t, packed_t>(v_weight[i]);
      elems[2 * i] = val.x * wval.x * rsqrt_val;
      elems[2 * i + 1] = val.y * wval.y * rsqrt_val;
    }

    // This thread owns elements [my_offset_in_dim, my_offset_in_dim + kElemsPerVec).
    // Because kHeadDim % kElemsPerVec == 0, this slice lies entirely within
    // a single head; offset_in_head is the slice's start within that head.
    const int my_offset_in_dim = threadIdx.x * kElemsPerVec;
    const int offset_in_head = my_offset_in_dim % kHeadDim;

    if (offset_in_head < kRopeDim) {
      const int64_t pos = static_cast<int64_t>(positions[token_id]);
      // cos_sin_cache stride is rope_dim float elements per token, laid out
      // as [cos[0..rope_dim/2-1], sin[0..rope_dim/2-1]] (FlashInfer layout).
      const float* cos_ptr = cos_sin_cache + pos * static_cast<int64_t>(kRopeDim);
      const float* sin_ptr = cos_ptr + kRopeDim / 2;

#pragma unroll
      for (int i = 0; i < kElemsPerVec; i += 2) {
        const int elem_idx_in_head = offset_in_head + i;
        if (elem_idx_in_head < kRopeDim) {
          const int half_idx = elem_idx_in_head / 2;
          const float cos_val = cos_ptr[half_idx];
          const float sin_val = sin_ptr[half_idx];
          const float x = elems[i];
          const float y = elems[i + 1];
          elems[i] = x * cos_val - y * sin_val;
          elems[i + 1] = y * cos_val + x * sin_val;
        }
      }
    }

#pragma unroll
    for (int i = 0; i < inner_loop; i++) {
      v_data[i] = device::cast<packed_t, fp32x2_t>(make_float2(elems[2 * i], elems[2 * i + 1]));
    }
    data[threadIdx.x] = v_data;
  }
}

template <int kHeadDim, int kRopeDim, typename DType>
struct QKNormAcrossHeadsRopeKernel {
  static_assert(std::is_same_v<DType, fp16_t> || std::is_same_v<DType, bf16_t>);
  static_assert(kHeadDim <= 256, "head_dim > 256 is not supported for warp-level across-heads RoPE");
  static_assert(kRopeDim <= kHeadDim, "rope_dim must not exceed head_dim");

  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView positions,
      float eps) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto D = SymbolicSize{"hidden_size"};
    auto R = SymbolicSize{"rope_dim"};
    auto device = SymbolicDevice{};
    auto id_type = SymbolicDType{};
    R.set_value(kRopeDim);
    device.set_options<kDLCUDA>();

    TensorMatcher({N, D})
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q);
    TensorMatcher({N, D})
        .with_strides({D, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k);
    TensorMatcher({D})
        .with_dtype<DType>()
        .with_device(device)
        .verify(q_weight);
    TensorMatcher({D})
        .with_dtype<DType>()
        .with_device(device)
        .verify(k_weight);
    TensorMatcher({-1, R}).with_dtype<float>().with_device(device).verify(cos_sin_cache);
    TensorMatcher({N}).with_dtype<int32_t, int64_t>(id_type).with_device(device).verify(positions);

    const int hidden_size = static_cast<int>(D.unwrap());
    host::RuntimeCheck(
        hidden_size % kHeadDim == 0, "hidden_size ", hidden_size, " must be divisible by head_dim ", kHeadDim);
    host::RuntimeCheck(hidden_size > 0 && hidden_size <= (device::kMaxVecBytes == 32 ? 12288 : 8192),
                       "hidden_size ", hidden_size, " out of supported range for across-heads RoPE fusion");

    const int elements_in_vec = device::kMaxVecBytes / sizeof(DType);
    const int vec_hidden_size = hidden_size / elements_in_vec;
    const uint threads = (vec_hidden_size + 31) / 32 * 32;

    host::RuntimeCheck(
        hidden_size % elements_in_vec == 0,
        "hidden_size ", hidden_size, " cannot align to elements_in_vec ", elements_in_vec);
    host::RuntimeCheck(
        kHeadDim % elements_in_vec == 0,
        "head_dim ", kHeadDim, " must be divisible by elements_in_vec ", elements_in_vec);
    host::RuntimeCheck(
        kRopeDim % elements_in_vec == 0,
        "rope_dim ", kRopeDim, " must be divisible by elements_in_vec ", elements_in_vec);

    const auto is_int32 = id_type.is_type<int32_t>();
    auto kernel_i32 = qknorm_across_heads_rope_kernel<DType, device::kMaxVecBytes, kHeadDim, kRopeDim, int32_t>;
    auto kernel_i64 = qknorm_across_heads_rope_kernel<DType, device::kMaxVecBytes, kHeadDim, kRopeDim, int64_t>;

    if (is_int32) {
      LaunchKernel(dim3(static_cast<uint>(N.unwrap()), 2), threads, device.unwrap())
          .enable_pdl(false)(
              kernel_i32,
              reinterpret_cast<DType*>(q.data_ptr()),
              reinterpret_cast<DType*>(k.data_ptr()),
              reinterpret_cast<DType*>(q_weight.data_ptr()),
              reinterpret_cast<DType*>(k_weight.data_ptr()),
              reinterpret_cast<const float*>(cos_sin_cache.data_ptr()),
              reinterpret_cast<const int32_t*>(positions.data_ptr()),
              vec_hidden_size,
              eps);
    } else {
      LaunchKernel(dim3(static_cast<uint>(N.unwrap()), 2), threads, device.unwrap())
          .enable_pdl(false)(
              kernel_i64,
              reinterpret_cast<DType*>(q.data_ptr()),
              reinterpret_cast<DType*>(k.data_ptr()),
              reinterpret_cast<DType*>(q_weight.data_ptr()),
              reinterpret_cast<DType*>(k_weight.data_ptr()),
              reinterpret_cast<const float*>(cos_sin_cache.data_ptr()),
              reinterpret_cast<const int64_t*>(positions.data_ptr()),
              vec_hidden_size,
              eps);
    }
  }
};

}  // namespace
