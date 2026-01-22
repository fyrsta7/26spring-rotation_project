      &builder, {97.5f, {3.13f, 3.14f}, {5.0f, 1.0f}, {-1.0f, 0.5f}}, {},
      error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantZeroElementC64s) {
  XlaBuilder builder(TestName());
  auto a = ConstantR1<complex64>(&builder, {});
  auto b = ConstantR1<complex64>(&builder, {});
  Add(a, b);

  ComputeAndCompareR1<complex64>(&builder, {}, {}, error_spec_);
}

XLA_TEST_F(ArrayElementwiseOpTest, AddTwoConstantU64s) {
  XlaBuilder b(TestName());

  std::vector<uint64_t> lhs{0xFFFFFFFF,
                            static_cast<uint64_t>(-1),
                            0,
                            0,
                            0x7FFFFFFFFFFFFFFFLL,
                            0x7FFFFFFFFFFFFFFLL,
                            0x8000000000000000ULL,
                            0x8000000000000000ULL,
                            1};
  Literal lhs_literal = LiteralUtil::CreateR1<uint64_t>({lhs});
  auto lhs_param = Parameter(&b, 0, lhs_literal.shape(), "lhs_param");
  std::unique_ptr<GlobalData> lhs_data =
      client_->TransferToServer(lhs_literal).value();

  std::vector<uint64_t> rhs{1,
                            0x7FFFFFFFFFFFFFFLL,
                            0x7FFFFFFFFFFFFFFFLL,
                            0x8000000000000000ULL,
                            0,
                            static_cast<uint64_t>(-1),
                            0,
                            1,
                            0x8000000000000000ULL};
  Literal rhs_literal = LiteralUtil::CreateR1<uint64_t>({rhs});
