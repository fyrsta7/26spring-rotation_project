/* static */ constexpr const char* const
    OptimizeDatasetOp::kOptimizationConfigs;
/* static */ constexpr const char* const OptimizeDatasetOp::kOptimizeDatasetV1;
/* static */ constexpr const char* const OptimizeDatasetOp::kOptimizeDatasetV2;

namespace {

constexpr char kOptimizerName[] = "tf_data_meta_optimizer";
constexpr char kOptimizers[] = "optimizers";
constexpr char kOptimizerConfigs[] = "optimizer_configs";

// A wrapper around `SelectOptimizations` responsible for configuring which
// tf.data experiments to apply.
std::vector<tstring> SelectOptimizationsHelper(
    const std::vector<tstring>& optimizations_enabled,
    const std::vector<tstring>& optimizations_disabled,
    const std::vector<tstring>& optimizations_default) {
  string job_name = port::JobName();
  // The map that stores the live experiment names and for how much percentage
  // of the Borg jobs, the experiments will be randomly turned on.
  // clang-format off
  absl::flat_hash_map<string, uint64> live_experiments = {
    {"enable_gradient_descent", 0},
    {"use_private_thread_pool", 1}
  };
  // clang-format on
  auto hash_func = [](const string& str) { return Hash64(str); };
  auto optimizations = SelectOptimizations(
      job_name, live_experiments, optimizations_enabled, optimizations_disabled,
      optimizations_default, hash_func);

  // Log and record the live experiments that will be applied.
  if (!job_name.empty() && !live_experiments.empty()) {
    VLOG(1) << "The input pipeline is subject to tf.data experiment. "
