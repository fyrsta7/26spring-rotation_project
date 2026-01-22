  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, std::min(vector[v], 6.0f));
  }
}

// Apply signbit to elements of a vector
void ApplySignbitToVector(const float* __restrict__ vector, int v_size,
                          float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::signbit(vector[v]);
  }
}

void UnpackDenseInt4IntoInt8(const int8_t* src_buffer, int num_elements,
                             int8_t* dst_buffer) {
  for (int i = 0; i < num_elements / 2; i++) {
    int8_t byte = src_buffer[i];
