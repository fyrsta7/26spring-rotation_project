          ") >= ", "(", trt_output_shape.d[1], ", ", trt_output_shape.d[2],
          ") for op ", node_def.name());
    }
    // Only add a padding layer if padding sizes are larger than 0
    if ((height_diff > 0) || (width_diff > 0)) {
      nvinfer1::DimsHW pre_padding(0, 0);
      nvinfer1::DimsHW post_padding(height_diff, width_diff);
      nvinfer1::IPaddingLayer* padding_layer =
          params->converter->network()->addPadding(*output_tensor, pre_padding,
                                                   post_padding);
      output_tensor = padding_layer->getOutput(0);
    }
  }
  // Restore transpose.
  if (need_transpose) {
    TF_RETURN_IF_ERROR(params->converter->TransposeTensor(
        output_tensor, {0, 2, 3, 1}, StrCat(node_def.name(), "_to_NHWC"),
        &output_tensor));
  }
  params->outputs->push_back(TRT_TensorOrWeights(output_tensor));
  return Status::OK();
}

Status ConvertTranspose(OpConverterParams* params) {
  const auto& inputs = params->inputs;
  TF_RETURN_IF_ERROR(
      CheckInputsWeights(*params, {{"x", false}, {"perm", true}}));
  TF_RETURN_IF_ERROR(AllowDataTypes(
      *params, {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}));
  // Get the permutation from weights.
  TRT_ShapedWeights weights = inputs.at(1).weights();
  const int* weights_ptr = static_cast<int*>(weights.GetValues());
  std::vector<int> perm(weights_ptr, weights_ptr + weights.count());

  // Verify the permutation.
  nvinfer1::ITensor* input_tensor = inputs.at(0).tensor();
  if (perm.size() - 1 != size_t(input_tensor->getDimensions().nbDims)) {
    return errors::InvalidArgument(
        "Rank of perm for transpose does not match with that of the input.");
