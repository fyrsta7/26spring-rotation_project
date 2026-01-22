#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_INTEGER_OPS_ADD_H_

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_integer_ops {

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const int8* input1_data, const int8* input2_data,
                           int8* output_data) {
  ruy::profiler::ScopeLabel label("AddElementwiseInt8/8bit");
  int i = 0;
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);

#ifdef USE_NEON
  const int8x16_t output_activation_min_vector =
      vdupq_n_s8(params.quantized_activation_min);
  const int8x16_t output_activation_max_vector =
      vdupq_n_s8(params.quantized_activation_max);

  const int input1_left_shift = params.left_shift + params.input1_shift;
  const int input2_left_shift = params.left_shift + params.input2_shift;
  const int32x4_t input1_left_dup = vdupq_n_s32(input1_left_shift);
  const int32x4_t input2_left_dup = vdupq_n_s32(input2_left_shift);

  for (; i <= size - 16; i += 16) {
    const int8x16_t input1_val_original = vld1q_s8(input1_data + i);
    const int8x16_t input2_val_original = vld1q_s8(input2_data + i);

    const int16x8_t input1_val_s16_high =
        vmovl_s8(vget_high_s8(input1_val_original));
    const int16x8_t input1_val_s16_low =
        vmovl_s8(vget_low_s8(input1_val_original));

    const int16x8_t input2_val_s16_high =
        vmovl_s8(vget_high_s8(input2_val_original));
    const int16x8_t input2_val_s16_low =
        vmovl_s8(vget_low_s8(input2_val_original));
    const int16x8_t input1_val_high =
        vaddq_s16(input1_val_s16_high, vdupq_n_s16(params.input1_offset));
    const int16x8_t input2_val_high =
        vaddq_s16(input2_val_s16_high, vdupq_n_s16(params.input2_offset));
    const int16x8_t input1_val_low =
        vaddq_s16(input1_val_s16_low, vdupq_n_s16(params.input1_offset));
    const int16x8_t input2_val_low =
        vaddq_s16(input2_val_s16_low, vdupq_n_s16(params.input2_offset));
    const int16x4_t input1_val_high_high = vget_high_s16(input1_val_high);
    const int16x4_t input1_val_high_low = vget_low_s16(input1_val_high);
    const int16x4_t input1_val_low_high = vget_high_s16(input1_val_low);
    const int16x4_t input1_val_low_low = vget_low_s16(input1_val_low);
    const int16x4_t input2_val_high_high = vget_high_s16(input2_val_high);
    const int16x4_t input2_val_high_low = vget_low_s16(input2_val_high);
    const int16x4_t input2_val_low_high = vget_high_s16(input2_val_low);
    const int16x4_t input2_val_low_low = vget_low_s16(input2_val_low);
    int32x4_t x111 = vmovl_s16(input1_val_low_low);
    int32x4_t x112 = vmovl_s16(input1_val_low_high);
    int32x4_t x121 = vmovl_s16(input1_val_high_low);
    int32x4_t x122 = vmovl_s16(input1_val_high_high);
    int32x4_t x211 = vmovl_s16(input2_val_low_low);
    int32x4_t x212 = vmovl_s16(input2_val_low_high);
    int32x4_t x221 = vmovl_s16(input2_val_high_low);
    int32x4_t x222 = vmovl_s16(input2_val_high_high);

    x111 = vshlq_s32(x111, input1_left_dup);
    x112 = vshlq_s32(x112, input1_left_dup);
    x121 = vshlq_s32(x121, input1_left_dup);
    x122 = vshlq_s32(x122, input1_left_dup);
    x211 = vshlq_s32(x211, input2_left_dup);
    x212 = vshlq_s32(x212, input2_left_dup);
    x221 = vshlq_s32(x221, input2_left_dup);
    x222 = vshlq_s32(x222, input2_left_dup);
    x111 = vqrdmulhq_n_s32(x111, params.input1_multiplier);
    x112 = vqrdmulhq_n_s32(x112, params.input1_multiplier);
    x121 = vqrdmulhq_n_s32(x121, params.input1_multiplier);
    x122 = vqrdmulhq_n_s32(x122, params.input1_multiplier);
    x211 = vqrdmulhq_n_s32(x211, params.input2_multiplier);
    x212 = vqrdmulhq_n_s32(x212, params.input2_multiplier);
    x221 = vqrdmulhq_n_s32(x221, params.input2_multiplier);
    x222 = vqrdmulhq_n_s32(x222, params.input2_multiplier);
    int32x4_t s11 = vaddq_s32(x111, x211);
    int32x4_t s12 = vaddq_s32(x112, x212);
    int32x4_t s21 = vaddq_s32(x121, x221);
    int32x4_t s22 = vaddq_s32(x122, x222);
    s11 = vqrdmulhq_n_s32(s11, params.output_multiplier);
    s12 = vqrdmulhq_n_s32(s12, params.output_multiplier);
    s21 = vqrdmulhq_n_s32(s21, params.output_multiplier);
    s22 = vqrdmulhq_n_s32(s22, params.output_multiplier);
    using gemmlowp::RoundingDivideByPOT;
    s11 = RoundingDivideByPOT(s11, -params.output_shift);
    s12 = RoundingDivideByPOT(s12, -params.output_shift);
    s21 = RoundingDivideByPOT(s21, -params.output_shift);
    s22 = RoundingDivideByPOT(s22, -params.output_shift);
    const int16x4_t s11_narrowed = vmovn_s32(s11);
    const int16x4_t s12_narrowed = vmovn_s32(s12);
    const int16x4_t s21_narrowed = vmovn_s32(s21);
    const int16x4_t s22_narrowed = vmovn_s32(s22);
    const int16x8_t s1 = vaddq_s16(vcombine_s16(s11_narrowed, s12_narrowed),
                                   vdupq_n_s16(params.output_offset));
    const int16x8_t s2 = vaddq_s16(vcombine_s16(s21_narrowed, s22_narrowed),
                                   vdupq_n_s16(params.output_offset));
    const int8x16_t s = vcombine_s8(vqmovn_s16(s1), vqmovn_s16(s2));

    const int8x16_t clamped =
        vmaxq_s8(output_activation_min_vector,
                 vminq_s8(output_activation_max_vector, s));
    vst1q_s8(output_data + i, clamped);
  }
#endif  // NEON

  for (; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32 scaled_input2_val =
