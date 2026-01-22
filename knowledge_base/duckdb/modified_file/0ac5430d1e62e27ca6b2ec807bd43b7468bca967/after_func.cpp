template <class SRC, class DST>
bool TryCastDecimalToFloatingPoint(SRC input, DST &result, uint8_t scale) {
	if (IsRepresentableExactly<SRC, DST>(input, DST(0.0)) || scale == 0) {
		// Fast path, integer is representable exaclty as a float/double
		result = Cast::Operation<SRC, DST>(input) / DST(NumericHelper::DOUBLE_POWERS_OF_TEN[scale]);
		return true;
	}
	auto power_of_ten = GetPowerOfTen(input, scale);
	result = Cast::Operation<SRC, DST>(input / power_of_ten) +
	         Cast::Operation<SRC, DST>(input % power_of_ten) / DST(NumericHelper::DOUBLE_POWERS_OF_TEN[scale]);
	return true;
}
