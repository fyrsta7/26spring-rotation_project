#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_replace.h"
#include "tensorflow/lite/delegates/gpu/common/task/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/relu.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace {
void UploadWeights(const DepthwiseConvolution2DAttributes& dw_attr,
                   const Convolution2DAttributes& conv_attr,
                   const GpuInfo& gpu_info, CalculationsPrecision precision,
                   GPUOperation* op) {
  int dw_dst_ch_aligned = AlignByN(dw_attr.weights.shape.i, 4);
  int dw_weights_count =
      dw_dst_ch_aligned * dw_attr.weights.shape.h * dw_attr.weights.shape.w;
  int conv_src_ch_aligned = AlignByN(conv_attr.weights.shape.i, 4);
  int conv_dst_ch_aligned = AlignByN(conv_attr.weights.shape.o, 4);
  int conv_weights_count = conv_src_ch_aligned * conv_dst_ch_aligned;
  std::vector<float> gpu_data;
  gpu_data.reserve(dw_dst_ch_aligned + dw_weights_count + conv_dst_ch_aligned +
                   conv_weights_count);
  // dw bias loading
  for (int i = 0; i < dw_dst_ch_aligned; ++i) {
    if (i < dw_attr.bias.shape.v) {
      gpu_data.push_back(dw_attr.bias.data[i]);
    } else {
      gpu_data.push_back(0.0f);
    }
  }
  // dw weights loading
  for (int d = 0; d < dw_dst_ch_aligned / 4; ++d) {
    for (int y = 0; y < dw_attr.weights.shape.h; ++y) {
      for (int x = 0; x < dw_attr.weights.shape.w; ++x) {
        for (int i = 0; i < 4; ++i) {
          const int d_ch = d * 4 + i;
          if (d_ch < dw_attr.weights.shape.i) {
            const int f_index =
                dw_attr.weights.shape.LinearIndex({0, y, x, d_ch});
            gpu_data.push_back(dw_attr.weights.data[f_index]);
          } else {
            gpu_data.push_back(0.0f);
          }
        }
      }
    }
  }
  // conv bias loading
  for (int i = 0; i < conv_dst_ch_aligned; ++i) {
    if (i < conv_attr.bias.shape.v) {
      gpu_data.push_back(conv_attr.bias.data[i]);
    } else {
      gpu_data.push_back(0.0f);
    }
  }
  // conv weights loading
  for (int d = 0; d < conv_dst_ch_aligned / 4; ++d) {
    for (int s = 0; s < conv_src_ch_aligned / 4; ++s) {
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          const int s_ch = s * 4 + j;
          const int d_ch = d * 4 + i;
          if (s_ch < conv_attr.weights.shape.i &&
              d_ch < conv_attr.weights.shape.o) {
            const int f_index =
                conv_attr.weights.shape.LinearIndex({d_ch, 0, 0, s_ch});
            gpu_data.push_back(conv_attr.weights.data[f_index]);
          } else {
            gpu_data.push_back(0.0f);
          }
        }
      }
    }
  }

  const bool fp32_weights = precision == CalculationsPrecision::F32;
  const int float_size = fp32_weights ? 4 : 2;
  BufferDescriptor desc;
  desc.element_type = fp32_weights ? DataType::FLOAT32 : DataType::FLOAT16;
  desc.element_size = 4;
  desc.memory_type =
      gpu_info.IsMali() ? MemoryType::GLOBAL : MemoryType::CONSTANT;
  desc.size = float_size * gpu_data.size();
