#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/storage_type_util.h"
#include "tensorflow/lite/delegates/gpu/common/task/weights_layout.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"

namespace tflite {
namespace gpu {

ConvolutionTransposed::ConvolutionTransposed(
    const OperationDef& definition, const ConvolutionTransposedAttributes& attr,
    const GpuInfo& gpu_info, bool weights_are_buffer)
    : GPUOperation(definition),
      stride_(attr.stride.w, attr.stride.h, 1, 1),
      block_size_(2, 2, 1, 2) {
  if (weights_are_buffer) {
    if (gpu_info.IsApple()) {
      weights_layout_ = WeightsLayout::kOSpatialIOGroupO4I4;
    } else {
      weights_layout_ = WeightsLayout::kOSpatialIOGroupI4O4;
    }
  } else {
    if (gpu_info.IsApple()) {
      weights_layout_ = WeightsLayout::k2DX4O4YIsSpatialIAndXIsOOGroupI4;
    } else {
      weights_layout_ = WeightsLayout::k2DX4I4YIsSpatialIAndXIsOOGroupO4;
    }
  }
  const bool is_f16 = definition.precision == CalculationsPrecision::F16;
  if (gpu_info.IsMali()) {
    if (gpu_info.mali_info.IsMidgard()) {
      block_size_ = is_f16 ? int4(2, 1, 1, 2) : int4(2, 1, 1, 1);
    } else {
      block_size_ = is_f16 ? int4(2, 2, 1, 2) : int4(2, 2, 1, 1);
    }
    compiler_options_.push_back(CompilerOptions::kClFastRelaxedMath);
  }
  const int dst_depth = DivideRoundUp(attr.weights.shape.o, 4);
  if (dst_depth == 1 || dst_depth == 3) {
    if (!gpu_info.IsMali()) {
      block_size_.y *= block_size_.w;
