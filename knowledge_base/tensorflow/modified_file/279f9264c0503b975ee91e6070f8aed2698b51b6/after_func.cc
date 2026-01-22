    // Interpolation is done using the fractional bit.
    uint32_t result = (ua << 9) + ut * (ub - ua);

    result = (input_data >=0) ? (result + (1 << 9)) :
                  ((1 << (16 + 9)) - result + (1 << 9) - 1);

    // Back to 16-bit.
    result >>= 10;

    *ptr_output_data = result;
  }
}

void EvalUsingLookupTableTanh16Bit(struct OpData* data, const TfLiteTensor* input,
                          TfLiteTensor* output) {

  const int size =
      MatchingFlatSize(GetTensorShape(input), GetTensorShape(output));

  const int16_t* ptr_input_data = GetTensorData<int16_t>(input);
  int16_t* ptr_output_data = GetTensorData<int16_t>(output);

  // We use the LUT for sigmoid and take into account, that
  // tanh(x) = 2*sigmoid(2*x) - 1
  for (int i=0; i < size; ++i, ptr_input_data++, ptr_output_data++) {

    int32_t input_data = *ptr_input_data;

    if (data->input_left_shift == 1) {
      input_data <<= 1;
    }

    // Scale by 3/4 to expand range [-8,8]->[-10.7,10.7].
    uint32_t abs_input_data = 3*abs(input_data);
    uint32_t uh = abs_input_data >> 8;
    int32_t result;

    if (uh >= 255) {
      // Saturate to maximum.
      result = 0xFFFF<<8;
    } else {

      uint32_t ua = data->table_zero[uh];
      uint32_t ub = data->table_zero[uh+1];

      uint8_t ut = abs_input_data & 0xFF;
