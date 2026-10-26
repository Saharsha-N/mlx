// Copyright © 2026 Apple Inc.

#include "mlx/backend/metal/kernels/expm1f.h"

template <
    typename T,
    typename AccT = float,
    int N_READS = CROSS_ENTROPY_N_READS>
[[kernel]] void cross_entropy(
    const device T* in,
    const device int32_t* targets,
    device float* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  in += gid * size_t(axis_size);

  constexpr int SIMD_SIZE = 32;

  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];

  int y_n = targets[gid];
  y_n = (y_n < 0) ? y_n + axis_size : y_n;
  AccT x_t = AccT(in[y_n]);

  AccT prevmax;
  AccT maxval = Limits<AccT>::finite_min;
  AccT normalizer = 0;
  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    AccT vals[N_READS];
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = AccT(in[offset + i]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        vals[i] =
            (offset + i < axis_size) ? AccT(in[offset + i]) : Limits<AccT>::min;
      }
    }
    // Leave out the target so that its e^0 = 1 does not absorb small terms.
    for (int i = 0; i < N_READS; i++) {
      vals[i] = (offset + i == y_n) ? Limits<AccT>::min : vals[i];
    }
    prevmax = maxval;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < vals[i]) ? vals[i] : maxval;
    }
    normalizer *= fast::exp(prevmax - maxval);
    for (int i = 0; i < N_READS; i++) {
      normalizer += fast::exp(vals[i] - maxval);
    }
  }

  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);

  uint n_simdgroups = ceildiv(lsize, uint(SIMD_SIZE));
  prevmax = maxval;
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = (simd_lane_id < n_simdgroups) ? local_max[simd_lane_id]
                                         : Limits<AccT>::finite_min;
  maxval = simd_max(maxval);
  normalizer *= fast::exp(prevmax - maxval);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer =
      (simd_lane_id < n_simdgroups) ? local_normalizer[simd_lane_id] : AccT(0);
  normalizer = simd_sum(normalizer);

  if (lid == 0) {
    // a is the logsumexp of the other logits minus the target logit, and the
    // loss is softplus(a).
    AccT gap = maxval - x_t;
    AccT a = isinf(maxval) ? gap : log(normalizer) + gap;
    out[gid] = float(a > 0 ? a + log1p(fast::exp(-a)) : log1p(fast::exp(a)));
  }
}

template <
    typename T,
    typename AccT = float,
    int N_READS = CROSS_ENTROPY_N_READS>
[[kernel]] void cross_entropy_vjp(
    const device T* in,
    const device int32_t* targets,
    const device float* loss,
    const device float* cotan,
    device T* out,
    constant int& axis_size,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]]) {
  size_t row_offset = gid * size_t(axis_size);
  in += row_offset;
  out += row_offset;

  int y_n = targets[gid];
  y_n = (y_n < 0) ? y_n + axis_size : y_n;
  AccT x_t = AccT(in[y_n]);
  AccT g = AccT(cotan[gid]);
  AccT l = AccT(loss[gid]);
  // p_t - 1 = e^-l - 1, without the cancellation.
  AccT d_t = expm1f(-l);

  // out aliases in when the input is donated, so every thread must read the
  // target column before any thread writes.
  threadgroup_barrier(mem_flags::mem_device);

  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        int col = offset + i;
        AccT p = fast::exp((AccT(in[col]) - x_t) - l);
        out[col] = T(g * ((col == y_n) ? d_t : p));
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        int col = offset + i;
        if (col < axis_size) {
          AccT p = fast::exp((AccT(in[col]) - x_t) - l);
          out[col] = T(g * ((col == y_n) ? d_t : p));
        }
      }
    }
  }
}
