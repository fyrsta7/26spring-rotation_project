// shape = [I, J, K]
// strides = [K*J, K, 1]
// void recurse(T* data, shape, strides, depth = 0) {
//   if(depth == shape.size) {
//     *data = ...
//   } else {
//     for(a = 0; a < shape[depth]; ++a) {
//       recurse(data, shape, strides, depth+1);
//       data += strides[depth];
//     }
//   }
// }
// ```
template <typename T>
void TransposeImpl(const int depth, const int dims, const int32_t* perm,
                   const T* input_data, const int* input_stride, T* output_data,
                   const int* output_stride, const int32_t* output_shape) {
  if (depth == dims - 1) {
    for (int i = 0; i < output_shape[depth]; ++i) {
