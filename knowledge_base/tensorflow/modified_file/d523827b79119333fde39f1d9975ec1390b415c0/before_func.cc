#include "tensorflow/lite/delegates/gpu/cl/kernels/elementwise.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/cl/storage_type_util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {
std::string GetOneInputCode(const OperationType& op_type,
                            CalculationsPrecision precision,
                            const std::string& input0) {
  std::string result;
  switch (op_type) {
    case OperationType::ABS:
      result = "$0 = fabs($0);\n";
      break;
    case OperationType::COS:
      result = "$0 = cos($0);\n";
      break;
    case OperationType::COPY:
      // No op as inout_value will be copied to dest automatically.
      result = "\n";
      break;
    case OperationType::ELU:
      result = "$0.x = $0.x < (FLT)(0.0f) ? exp($0.x) - (FLT)(1.0f) : $0.x;\n";
      result += "$0.y = $0.y < (FLT)(0.0f) ? exp($0.y) - (FLT)(1.0f) : $0.y;\n";
      result += "$0.z = $0.z < (FLT)(0.0f) ? exp($0.z) - (FLT)(1.0f) : $0.z;\n";
      result += "$0.w = $0.w < (FLT)(0.0f) ? exp($0.w) - (FLT)(1.0f) : $0.w;\n";
      break;
    case OperationType::EXP:
      result = "$0 = exp($0);\n";
      break;
    case OperationType::HARD_SWISH:
      result =
          "$0 *= clamp($0 * (FLT)(0.16666667f) + (FLT)(0.5f), (FLT4)(0.0f), "
          "(FLT4)(1.0f));\n";
      break;
    case OperationType::LOG:
      result = "$0 = log($0);\n";
      break;
    case OperationType::RSQRT:
      result = "$0 = (FLT4)(1.0f) / sqrt($0);\n";
      break;
    case OperationType::SIGMOID:
      if (precision != CalculationsPrecision::F32) {
        result =
            "$0.x = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.x))));\n";
        result +=
            "$0.y = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.y))));\n";
        result +=
            "$0.z = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.z))));\n";
        result +=
            "$0.w = convert_half(native_recip(1.0f + "
            "native_exp(convert_float(-$0.w))));\n";
      } else {
        result = "$0 = (FLT4)(1.0f) / ((FLT4)(1.0f) + exp(-($0)));\n";
      }
      break;
    case OperationType::SIN:
      result = "$0 = sin($0);\n";
      break;
    case OperationType::SQRT:
