
#include <algorithm>
#include <functional>

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace gml_st {

GmlStCPUTilingOptions getDefaultCPUPipelineOptions(StringRef cpuName,
                                                   int64_t statsDetailLevel) {
  GmlStCPUTilingOptions opts;
  opts.vectorSize = 8;
