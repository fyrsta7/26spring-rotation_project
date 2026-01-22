  ruy_mul_params->set_multiplier_exponent(params.multiplier_exponent);
  ruy_mul_params->set_multiplier_fixedpoint_perchannel(
      params.multiplier_fixedpoint_perchannel);
  ruy_mul_params->set_multiplier_exponent_perchannel(
      params.multiplier_exponent_perchannel);
  ruy_mul_params->set_bias(params.bias);
  ruy_mul_params->set_clamp_min(params.clamp_min);
  ruy_mul_params->set_clamp_max(params.clamp_max);
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplUsingRuy {
  static void Run(
      const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
      const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
      const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
      CpuBackendContext* context) {
