static bool rangeContainsTokenLocWithGeneratedSource(
    SourceManager &sourceMgr, SourceRange parentRange, SourceLoc childLoc) {
  if (sourceMgr.rangeContainsTokenLoc(parentRange, childLoc))
    return true;

  auto childBuffer = sourceMgr.findBufferContainingLoc(childLoc);
  while (!sourceMgr.rangeContainsTokenLoc(
      sourceMgr.getRangeForBuffer(childBuffer), parentRange.Start)) {
    auto *info = sourceMgr.getGeneratedSourceInfo(childBuffer);
    if (!info)
      return false;

    childLoc = info->originalSourceRange.getStart();
    if (childLoc.isInvalid())
      return false;

    childBuffer = sourceMgr.findBufferContainingLoc(childLoc);
  }

  return sourceMgr.rangeContainsTokenLoc(parentRange, childLoc);
}
