    Type dtype;
    if (handle.dtype() != tensorflow::DT_INVALID)
      TF_RETURN_IF_ERROR(ConvertDataType(handle.dtype(), builder, &dtype));
    TF_ASSIGN_OR_RETURN(
        Attribute shape,
        ConvertTensorShapeProto(handle.shape(), builder.getContext()));

    dtype_and_shape.push_back(
        builder.getArrayAttr({TypeAttr::get(dtype), shape}));
  }
  return builder.getArrayAttr(dtype_and_shape);
}

tensorflow::StatusOr<OwningModuleRef> ImportGraphDefToMlir(
