// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include <algorithm>
#include <functional>
#include <future>
#include <string>
#include <vector>

#include "gsl/span"
#include "gsl/gsl_algorithm"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#ifdef USE_EIGEN_THREADPOOL
#include <unsupported/Eigen/CXX11/ThreadPool>
#else
#include "core/common/task_thread_pool.h"
#endif

namespace onnxruntime {
class Tensor;
class OpKernelContext;

namespace rnn {
namespace detail {

enum Direction {
  kForward = 0,
  kReverse = 1,
  kBidirectional = 2
};

inline Direction MakeDirection(const std::string& direction) {
  if (direction == "forward") {
    return kForward;
  } else if (direction == "reverse") {
    return kReverse;
  } else if (direction == "bidirectional") {
    return kBidirectional;
  } else {
    ORT_THROW("Invalid 'direction' argument of '", direction,
              "'. Must be one of 'forward', 'reverse', or 'bidirectional'.");
  }
}

/** Allocate a unique_ptr using allocator_, and return a span to the allocated memory so usage is safe
@param allocator IAllocator to use for the allocation.
@param size Allocation size. Number of elements of type TAlloc, or total size if TAlloc is 'void'.
@param unique_ptr unique_ptr that will control the lifetime of the allocated memory.
@param fill If true, fill the allocated memory with fill_value.
@param fill_value Value to use if 'fill' is true.
@returns A span to provide bounds checked access to the allocated memory.
*/
template <typename TAlloc>
gsl::span<TAlloc> Allocate(std::shared_ptr<IAllocator> allocator,
                           size_t size,
                           IAllocatorUniquePtr<TAlloc>& unique_ptr,
                           bool fill = false, TAlloc fill_value = TAlloc{}) {
  unique_ptr = IAllocator::MakeUniquePtr<TAlloc>(allocator, size);
  auto span = gsl::make_span(unique_ptr.get(), size);

  if (fill) {
    // Do't use span.begin() it will cause performance issue and stop compiler to optimize the code
    std::fill_n(unique_ptr.get(), size, fill_value);
  }

  return span;
}

// validate the common inputs to RNN, LSTM and GRU operators
Status ValidateCommonRnnInputs(const Tensor& X,
                               const Tensor& W,
                               const Tensor& R,
                               const Tensor* B,
                               int WRB_dim_1_multipler,  // multiplier used with hidden_size for W, R and B inputs
                               const Tensor* sequence_lens,
                               const Tensor* initial_h,
                               int64_t num_directions,
                               int64_t hidden_size);

/// Copy an input array repeatedly to an output array
/// @param input_begin Beginning of input
/// @param input_end End of input
/// @param output Output iterator
/// @param repetitions Number of times to repeat copy. Assumes output is sufficiently sized.
/// @returns Position of output iterator after copy is completed
template <typename TInIter, typename TOutIter>
TOutIter RepeatVectorToConstructArray(TInIter input_begin,
                                      TInIter input_end,
                                      TOutIter output,
                                      int64_t repetitions) {
  for (int64_t i = 0; i < repetitions; i++) {
    output = std::copy(input_begin, input_end, output);
  }

  return output;
}

// reverse an LSTM or GRU sequence which has shape [seq_length, batch_size, hidden_size]
// and output to shape [seq_length, num_directions, batch_size, hidden_size]
template <typename T>
void ReverseSequence(gsl::span<const T> inputs,
                     gsl::span<T> inputs_reverse,
                     gsl::span<const int> sequence_lengths,
                     const int max_sequence_length,
                     const int batch_size,
                     const int input_size,
                     const int num_directions) {
  for (int i = 0; i < batch_size; i++) {
    int seq_len = sequence_lengths[i];

    if (seq_len == 0)
      continue;
#ifdef USE_OPENMP
// Parallel execute the loop.
#pragma omp parallel for
#endif
    for (int j = 0; j < seq_len; j++) {
      gsl::span<const T> src = inputs.subspan(j * batch_size * input_size + i * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(num_directions * (seq_len - j - 1) * batch_size * input_size + i * input_size, input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }

#ifdef USE_OPENMP
// Parallel execute the loop.
#pragma omp parallel for
#endif
    for (int j = seq_len; j < max_sequence_length; j++) {
      gsl::span<const T> src = inputs.subspan(j * batch_size * input_size + i * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(num_directions * j * batch_size * input_size + i * input_size, input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }
  }
}

// A has size M x K, B has size N x K (transposed), and C has size M x N
// We check that A, B and C are large enough before calling the lower level GEMM implementation
template <typename TSpanAIter, typename TSpanBIter, typename TSpanCIter>
void ComputeGemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 TSpanAIter A,
                 TSpanAIter A_end,
                 const int lda,
                 TSpanBIter B,
                 TSpanBIter B_end,
                 const int ldb,
                 const float beta,
                 TSpanCIter C,
                 TSpanCIter C_end,
                 const int ldc) {
  // validate all the inputs
  // need to use the lda/ldb/ldc strides which should be >= the columns for the span
  ORT_ENFORCE(lda >= K && ldb >= K && ldc >= N);
  ORT_ENFORCE(A + (M * lda - (lda - K)) <= A_end);
  ORT_ENFORCE(B + (N * ldb - (ldb - K)) <= B_end);
  ORT_ENFORCE(C + (M * ldc - (ldc - N)) <= C_end);

  ::onnxruntime::math::GemmEx<float, CPUMathUtil>(
      CblasNoTrans, CblasTrans,
      M, N, K, alpha,
      &*A, lda,
      &*B, ldb, beta,
      &*C, ldc, &CPUMathUtil::Instance());
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
const T* SafeRawConstPointer(typename gsl::span<T>::const_iterator cur,
                             typename gsl::span<T>::const_iterator end,
                             size_t size) {
  ORT_ENFORCE(cur + size <= end);
  return &*cur;
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
const T* SafeRawConstPointer(gsl::span<T> span, size_t offset, size_t size) {
  ORT_ENFORCE(offset + size <= size_t(span.size()));
  return span.data();
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
T* SafeRawPointer(typename gsl::span<T>::iterator cur,
                  typename gsl::span<T>::iterator end,
                  size_t size) {
  ORT_ENFORCE(cur + size <= end);
  return &*cur;
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
T* SafeRawPointer(typename gsl::span<T> span, size_t offset, size_t size) {
  ORT_ENFORCE(offset + size <= size_t(span.size()));
  return span.data() + offset;
}

template <typename TLambda>
void ExecuteLambdaInParallel(const std::string& name, TLambda lambda, int max, int step,
#ifdef USE_EIGEN_THREADPOOL
                             Eigen::NonBlockingThreadPool& ttp,
#else
                             TaskThreadPool& ttp,
#endif
                             const ::onnxruntime::logging::Logger& logger) {
  // #define NOTHREADS to execute the lambdas directly and in order if you need to do that to debug

#ifdef NOTHREADS
  ORT_UNUSED_PARAMETER(ttp);
  ORT_UNUSED_PARAMETER(logger);

  for (int i = 0; i < max; i += step) {
    (void)name;
    std::bind(lambda, i)();
  }
#else

#ifdef USE_EIGEN_THREADPOOL
  ORT_UNUSED_PARAMETER(name);
  ORT_UNUSED_PARAMETER(logger);

  std::atomic<int> done(0);
  for (int i = 0; i < max; i += step) {
    ttp.Schedule([lambda, i, &done]() {
      lambda(i);
      ++done;
    });
  }

  int totalTasks = (int)max / (step > 0 ? step : 1) + (max % step > 0 ? 1 : 0);
  while (done != totalTasks) {
  }
#else
  std::vector<std::future<void> > task_results{};
  task_results.reserve(static_cast<size_t>(std::ceil(max / step)));

  for (int i = 0; i < max; i += step) {
    std::packaged_task<void()> task{std::bind(lambda, i)};
    task_results.emplace_back(task.get_future());
    ttp.RunTask(std::move(task));
  }
  try {
    // wait for all and propagate any exceptions
    for (auto& future : task_results)
      future.get();
  } catch (const std::exception& ex) {
    LOGS(logger, ERROR) << name << " - exception running tasks: " << ex.what();
    throw;
  }
#endif  // else part of #ifdef USE_EIGEN_THREADPOOLs
#endif  // else part of #ifdef NOTHREADS
}

void DumpMatrixImpl(const std::string& name, const float* src, int row, int col,
                    int offset = 0, int col_width = -1);

// Helper class to wrap the processing of the activation funcs and any alpha/beta values.
// The alpha/beta values are consumed in the order of the activation funcs. once they run out
// defaults will be used as needed.
// The Entries property contains the normalized function names and the alpha/beta value to use.
class ActivationFuncs {
 public:
  struct Entry {
    const std::string name;
    const float alpha;
    const float beta;
  };

  ActivationFuncs() = default;

  ActivationFuncs(const std::vector<std::string>& funcs,
                  const std::vector<float>& alphas,
                  const std::vector<float>& betas);

  const std::vector<Entry>& Entries() const {
    return entries_;
  }

 private:
  std::vector<Entry> entries_;
};

namespace deepcpu {

using AddBiasIntoFuncPtr = void (*)(const float*, float*, const int);
using ClipWithBiasFuncPtr = void (*)(const float, const float*, float*, const int);
using ActivationFuncPtr = void (*)(float*, const int, const float, const float);
using ActivationFuncBPtr = void (*)(const float*, float*, const int, const float, const float);
using LstmMergeGatesFuncPtr = void (*)(const float*, float*, const float*, float*, const int, const float, const float);
using GruResetGateFuncPtr = void (*)(const float*, float*, float*, const int, const float, const float);
using GruOutputGateFuncPtr = void (*)(float*, const float*, const float*, float*, const int, const float, const float);

ActivationFuncPtr ActivationFuncByName(const std::string& func);
LstmMergeGatesFuncPtr LstmMergeGatesFuncByName(const std::string& func);
GruResetGateFuncPtr GruResetGateFuncByName(const std::string& func);
GruOutputGateFuncPtr GruOutputGateFuncByName(const std::string& func);

void add_bias_into_ignore(const float* ignored, float* pd, const int c);
void add_bias_into(const float* ps, float* pd, const int c);
void clip(const float b, float* pd, const int c);
void clip_add_bias(const float b, const float* pb, float* pd, const int c);
void clip_ignore_bias(const float b, const float* pb, float* pd, const int c);
void sigmoid_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, const float alpha, const float beta);
void tanh_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, const float alpha, const float beta);
void relu_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, const float alpha, const float beta);
void sigmoid_exact_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, const float alpha, const float beta);
void tanh_exact_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, const float alpha, const float beta);
void sigmoid(float* pd, int c, const float alpha, const float beta);
void tanh(float* pd, int c, const float alpha, const float beta);
void relu(float* pd, int c, const float alpha, const float beta);
void sigmoid_exact(float* pd, int c, const float alpha, const float beta);
void tanh_exact(float* pd, int c, const float alpha, const float beta);
void merge_lstm_gates_to_memory(const float* pprev, const float* pi, const float* pf, const float* pg, float* pcurr, const int c);
void gru_reset_gate_tanh(const float* ps1, float* ps2, float* pd, const int c, const float alpha, const float beta);
void gru_reset_gate_sigmoid(const float* ps1, float* ps2, float* pd, const int c, const float alpha, const float beta);
void gru_reset_gate_relu(const float* ps1, float* ps2, float* pd, const int c, const float alpha, const float beta);
void gru_output_gate_tanh(float* ph, const float* pz, const float* ps, float* po, const int c, const float alpha, const float beta);
void gru_output_gate_sigmoid(float* ph, const float* pz, const float* ps, float* po, const int c, const float alpha, const float beta);
void gru_output_gate_relu(float* ph, const float* pz, const float* ps, float* po, const int c, const float alpha, const float beta);

inline void elementwise_product(const float* op1, const float* op2, float* dest, const int size) {
  for (int i = 0; i < size; i++)
    dest[i] += op1[i] * op2[i];
}

inline void elementwise_sum1(const float* src, float* dest, const int size) {
  for (int i = 0; i < size; i++)
    dest[i] += src[i];
}

inline void elementwise_sum2(const float* src1, const float* src2, float* dest, const int size) {
  for (int i = 0; i < size; i++)
    dest[i] += src1[i] + src2[i];
}

}  // namespace deepcpu
}  // namespace detail
}  // namespace rnn
}  // namespace onnxruntime
