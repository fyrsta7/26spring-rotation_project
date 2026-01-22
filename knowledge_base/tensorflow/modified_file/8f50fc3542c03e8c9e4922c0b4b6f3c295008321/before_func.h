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
    ruy::Matrix<LhsScalar> ruy_lhs;
    ruy::Matrix<RhsScalar> ruy_rhs;
    ruy::Matrix<DstScalar> ruy_dst;
    MakeRuyMatrix(lhs_params, lhs_data, &ruy_lhs);
    MakeRuyMatrix(rhs_params, rhs_data, &ruy_rhs);
    MakeRuyMatrix(dst_params, dst_data, &ruy_dst);

    ruy::MulParams<AccumScalar, DstScalar> ruy_mul_params;
    MakeRuyMulParams(params, &ruy_mul_params);

// If Ruy is not selected intentionally (TFLITE_WITH_RUY not defined)
