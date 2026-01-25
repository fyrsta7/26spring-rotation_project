absl::Status ClusterImplBase::parseDropOverloadConfig(
    const envoy::config::endpoint::v3::ClusterLoadAssignment& cluster_load_assignment) {
  // Default drop_overload_ to zero.
  drop_overload_ = UnitFloat(0);

  if (!cluster_load_assignment.has_policy()) {
    return absl::OkStatus();
  }
  const auto& policy = cluster_load_assignment.policy();
  if (policy.drop_overloads().size() == 0) {
    return absl::OkStatus();
  }
  if (policy.drop_overloads().size() > kDropOverloadSize) {
    return absl::InvalidArgumentError(
        fmt::format("Cluster drop_overloads config has {} categories. Envoy only support one.",
                    policy.drop_overloads().size()));
  }

  const auto& drop_percentage = policy.drop_overloads(0).drop_percentage();
  float denominator = 100;
  switch (drop_percentage.denominator()) {
  case envoy::type::v3::FractionalPercent::HUNDRED:
    denominator = 100;
    break;
  case envoy::type::v3::FractionalPercent::TEN_THOUSAND:
    denominator = 10000;
    break;
  case envoy::type::v3::FractionalPercent::MILLION:
    denominator = 1000000;
    break;
  default:
    return absl::InvalidArgumentError(fmt::format(
        "Cluster drop_overloads config denominator setting is invalid : {}. Valid range 0~2.",
        drop_percentage.denominator()));
  }
  drop_overload_ = UnitFloat(float(drop_percentage.numerator()) / (denominator));
  return absl::OkStatus();
}