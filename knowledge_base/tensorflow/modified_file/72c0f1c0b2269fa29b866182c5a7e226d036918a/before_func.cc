#include <utility>

#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/tensor_desc.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/mean_stddev_normalization.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/conv_pointwise.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/dw7x7_conv2to6_concat_conv8to8.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/fc_fc_add.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/special/thin_pointwise_fuser.h"

namespace tflite {
namespace gpu {
absl::Status GPUSubgraphFromGraph(
    const ModelHints& hints, const GpuInfo& gpu_info,
    CalculationsPrecision precision, const GraphFloat32& graph,
    NodeId first_node_id,
    const std::map<ValueId, TensorDescriptor>& tensor_descriptors,
    std::set<NodeId>* consumed_nodes, GPUOperationsSubgraph* gpu_subgraph) {
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryDW7x7Conv2To6ConcatConv8to8(gpu_info, precision, graph, first_node_id,
                                     tensor_descriptors, consumed_nodes,
                                     gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryThinPointwiseFuser(gpu_info, precision, graph, first_node_id,
                            tensor_descriptors, consumed_nodes, gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryFCFCAdd(gpu_info, precision, graph, first_node_id, tensor_descriptors,
                 consumed_nodes, gpu_subgraph)
          .ok()) {
    return absl::OkStatus();
  }
  if (hints.Check(ModelHints::kAllowSpecialKernels) &&
      TryFusedPointwiseConv(graph, first_node_id, precision, tensor_descriptors,
                            consumed_nodes, gpu_subgraph)
