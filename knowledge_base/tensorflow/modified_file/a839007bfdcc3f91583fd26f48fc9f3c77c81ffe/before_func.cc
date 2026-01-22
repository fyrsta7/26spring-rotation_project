#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace kernel_gen {
namespace {

using tensorflow::Status;
using xla::InternalError;
using xla::StatusOr;

constexpr llvm::StringRef kGpuBinaryAttrName = "gpu.binary";

Status LowerTFtoGPU(mlir::ModuleOp module, bool gpu_binary_only,
                    llvm::ArrayRef<uint32_t> tile_sizes,
                    llvm::ArrayRef<uint32_t> unroll_factors,
                    bool embed_memref_prints) {
  mlir::PassManager pm(module.getContext());
  applyTensorflowAndCLOptions(pm);

  if (gpu_binary_only) {
    pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeTFPass(
        /*allow_partial_conversion=*/false, /*legalize_chlo=*/true));
    pm.addNestedPass<mlir::FuncOp>(
        mlir::kernel_gen::transforms::CreateMaterializeBroadcastsPass());
    pm.addNestedPass<mlir::FuncOp>(
        mlir::kernel_gen::transforms::CreateUnfuseBatchNormPass());
    pm.addPass(mlir::mhlo::createLegalizeToLhloPass());
    // Moving `AllocOp`s and inserting missing `DeallocOp`s
    pm.addNestedPass<mlir::FuncOp>(::mlir::createBufferHoistingPass());
    pm.addNestedPass<mlir::FuncOp>(::mlir::createBufferDeallocationPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCopyRemovalPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::kernel_gen::transforms::CreateShapeToDescriptorsPass());
  } else {
    pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createLegalizeTFPass(
        /*allow_partial_conversion=*/false, /*legalize_chlo=*/false));
    pm.addNestedPass<mlir::FuncOp>(mlir::createTransformUnrankedHloPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::mhlo::createChloLegalizeToHloPass());
    pm.addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::kernel_gen::transforms::CreateShapeToDescriptorsPass());
    // Clean up the IR created above. In particular, operations on descriptors
    // are simplified here.
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::kernel_gen::transforms::CreateBufferizePass());
    pm.addNestedPass<mlir::FuncOp>(
        mlir::kernel_gen::transforms::CreateParallelLoopsToSequential());
  }

  // Clean up the IR for further processing.
  pm.addPass(mlir::createCanonicalizerPass());
  // We have to anticipate later unrolling in tiling to make sure that we get
  // the requested tiling after unrolling. Compute the new tiling here if
  // needed.
  llvm::SmallVector<unsigned, 4> tiling_for_unrolling;
  llvm::SmallVector<int64_t, 4> as_int64;
  if (!unroll_factors.empty()) {
    tiling_for_unrolling.reserve(tile_sizes.size());
    for (auto pair : llvm::zip(tile_sizes, unroll_factors)) {
      tiling_for_unrolling.push_back(std::get<0>(pair) * std::get<1>(pair));
      as_int64.push_back(std::get<1>(pair));
    }
  } else {
    tiling_for_unrolling.append(tile_sizes.begin(), tile_sizes.end());
  }
  // Transform LHLO operations to LinAlg.
  pm.addNestedPass<mlir::FuncOp>(
      ::mlir::lmhlo::createLegalizeLhloToLinalgPass());
  // Fuse linalg operations.
  pm.addNestedPass<mlir::FuncOp>(::mlir::lmhlo::createLhloFuseLinalgPass(
      /*use_parallel_loops=*/true, tiling_for_unrolling));
  // Transform the Linalg operations inside of the loop nest into parallel
  // loops.
  pm.addNestedPass<mlir::FuncOp>(
      ::mlir::createConvertLinalgToParallelLoopsPass());
  // Canonicalize the code to simplify index computations. This is needed so
  // that loop bounds have the same value.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Fuse the inner-most loops.
  pm.addNestedPass<mlir::FuncOp>(
      xla::mlir_gpu::createFuseInnerParallelLoopsPass());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Forward stores to buffers to loads.
  pm.addNestedPass<mlir::FuncOp>(xla::mlir_gpu::createStoreForwardingPass());
  // Remove now unused temporary buffers.
  pm.addNestedPass<mlir::FuncOp>(
      xla::mlir_gpu::createDeadTempBufferRemovalPass());
  if (!unroll_factors.empty()) {
    pm.addNestedPass<mlir::FuncOp>(
        ::mlir::createParallelLoopTilingPass(as_int64));
  }
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Greedily map the remaining loop to GPU hardware dimensions.
  pm.addNestedPass<::mlir::FuncOp>(xla::mlir_gpu::createMapParallelLoopsPass());
  // Apply the mapping.
  pm.addNestedPass<::mlir::FuncOp>(mlir::createParallelLoopToGpuPass());

  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Make loops with min bounds into a conditional plus static bounds.
  // Only do this if we unrolled in the first place.
  if (!unroll_factors.empty()) {
    pm.addNestedPass<::mlir::FuncOp>(mlir::createForLoopSpecializationPass());
  }
  // Approximate Tanh using standard operations.
  pm.addNestedPass<::mlir::FuncOp>(
      ::mlir::mhlo::createLegalizeTrigonometricToApproximationPass());
  // Take launches to launches with kernels.
  pm.addPass(::mlir::createGpuKernelOutliningPass());

  if (gpu_binary_only) {
    // Make kernel signature deterministic so that we can call it externally.
    pm.addNestedPass<::mlir::FuncOp>(
        xla::mlir_gpu::createRewriteKernelSignaturePass());
  }
  pm.addPass(::mlir::createLowerAffinePass());
  // Map allocs, asserts, etc. to the tensorflow framework.
  pm.addPass(mlir::kernel_gen::tf_framework::CreateEmbedTFFrameworkPass());
  // Constraints are removed as late as possible and before lowering to CFG.
