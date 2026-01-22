Node* WasmGraphBuilder::LoadTransformBigEndian(
    MachineType memtype, wasm::LoadTransformationKind transform, Node* value) {
  Node* result;
  LoadTransformation transformation = GetLoadTransformation(memtype, transform);

  switch (transformation) {
    case LoadTransformation::kS8x16LoadSplat: {
      result = graph()->NewNode(mcgraph()->machine()->I8x16Splat(), value);
      break;
    }
    case LoadTransformation::kS16x8LoadSplat: {
      result = graph()->NewNode(mcgraph()->machine()->I16x8Splat(), value);
      break;
    }
    case LoadTransformation::kS32x4LoadSplat: {
      result = graph()->NewNode(mcgraph()->machine()->I32x4Splat(), value);
      break;
    }
    case LoadTransformation::kS64x2LoadSplat: {
      result = graph()->NewNode(mcgraph()->machine()->I64x2Splat(), value);
      break;
    }
    default:
      UNREACHABLE();
  }

  return result;
}
