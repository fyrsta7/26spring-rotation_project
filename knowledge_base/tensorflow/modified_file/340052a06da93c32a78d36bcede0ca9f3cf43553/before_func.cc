    for (int i = start; i < end; ++i) {
      output = reducer(output, input_data[i]);
    }
  }

 private:
  EvalData<T>* eval_data;
  int start;
  int end;
};

// Apply reduce operation using the 'reducer' function on all of 'input_data'.
// and reduce all to single element.
template <typename T>
void ReduceAllDims(const T* input_data, const int* input_dims,
                   const int input_num_dims, T* output_data, T init_value,
                   T reducer(const T current, const T in),
                   TfLiteContext* context) {
  EvalData<T> eval_data;
  eval_data.reduce_func = reducer;
  eval_data.input_data = input_data;
  eval_data.output = init_value;

  int num_elems = 1;
  for (int i = 0; i < input_num_dims; ++i) {
    num_elems *= input_dims[i];
  }

  // Fetch backend context and number of threads.
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  const int thread_count = cpu_backend_context->max_num_threads();

  std::vector<ReduceWorkerTask<T>> tasks;
  std::vector<EvalData<T>> data;
  tasks.reserve(thread_count);
  data.reserve(thread_count);
  int start = 0;
  for (int i = 0; i < thread_count; ++i) {
    data.push_back(eval_data);
