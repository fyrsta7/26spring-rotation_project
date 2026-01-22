static Size getFixedBufferSize(IRGenModule &IGM) {
  return 3 * IGM.getPointerSize();
}
