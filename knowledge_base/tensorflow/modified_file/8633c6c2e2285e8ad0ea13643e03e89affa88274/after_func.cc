
Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);
TF_Tensor* TF_TensorFromTensor(const tensorflow::Tensor& src, Status* status);

Status NdarrayToTensor(PyObject* obj, Tensor* ret) {
  Safe_TF_TensorPtr tf_tensor = make_safe(static_cast<TF_Tensor*>(nullptr));
  Status s = NdarrayToTensor(nullptr /*ctx*/, obj, &tf_tensor);
  if (!s.ok()) {
