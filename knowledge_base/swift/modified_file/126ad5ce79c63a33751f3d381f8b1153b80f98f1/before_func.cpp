static bool rangeContainsTokenLocWithGeneratedSource(
    SourceManager &sourceMgr, SourceRange parentRange, SourceLoc childLoc) {
  auto parentBuffer = sourceMgr.findBufferContainingLoc(parentRange.Start);
  auto childBuffer = sourceMgr.findBufferContainingLoc(childLoc);
  while (parentBuffer != childBuffer) {
    auto info = sourceMgr.getGeneratedSourceInfo(childBuffer);
    if (!info)
      return false;

    childLoc = info->originalSourceRange.getStart();
    if (childLoc.isInvalid())
      return false;

    childBuffer = sourceMgr.findBufferContainingLoc(childLoc);
  }

  return sourceMgr.rangeContainsTokenLoc(parentRange, childLoc);
}
