const LineColumnRange &
CapturedFixItInfo::getLineColumnRange(const SourceManager &SM,
                                      unsigned BufferID) const {
  if (LineColRange.StartLine != 0) {
    // Already computed.
    return LineColRange;
  }

  auto SrcRange = FixIt.getRange();

  std::tie(LineColRange.StartLine, LineColRange.StartCol) =
      SM.getPresumedLineAndColumnForLoc(SrcRange.getStart(), BufferID);
  std::tie(LineColRange.EndLine, LineColRange.EndCol) =
      SM.getPresumedLineAndColumnForLoc(SrcRange.getEnd(), BufferID);

  return LineColRange;
}
