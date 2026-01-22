Node* WasmGraphBuilder::LoadTransformBigEndian(
    MachineType memtype, wasm::LoadTransformationKind transform, Node* value) {
  Node* result;
  LoadTransformation transformation = GetLoadTransformation(memtype, transform);

  switch (transformation) {
    case LoadTransformation::kS8x16LoadSplat: {
      result = graph()->NewNode(mcgraph()->machine()->I8x16Splat(), value);
      break;
    }
    case LoadTransformation::kI16x8Load8x8S:
    case LoadTransformation::kI16x8Load8x8U:
    case LoadTransformation::kS16x8LoadSplat: {
      result = graph()->NewNode(mcgraph()->machine()->I16x8Splat(), value);
      break;
    }
    case LoadTransformation::kI32x4Load16x4S:
    case LoadTransformation::kI32x4Load16x4U:
    case LoadTransformation::kS32x4LoadSplat: {
      result = graph()->NewNode(mcgraph()->machine()->I32x4Splat(), value);
      break;
    }
    case LoadTransformation::kI64x2Load32x2S:
    case LoadTransformation::kI64x2Load32x2U:
    case LoadTransformation::kS64x2LoadSplat: {
      result = graph()->NewNode(mcgraph()->machine()->I64x2Splat(), value);
      break;
    }
    default:
      UNREACHABLE();
  }

  return result;
}
