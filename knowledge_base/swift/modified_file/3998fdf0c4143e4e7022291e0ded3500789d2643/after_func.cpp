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

  // We don't have to compute much if the end location is on the same line.
  if (SrcRange.getByteLength() == 0) {
    LineColRange.EndLine = LineColRange.StartLine;
    LineColRange.EndCol = LineColRange.StartCol;
  } else if (SM.extractText(SrcRange, BufferID).find_first_of("\n\r") ==
             StringRef::npos) {
    LineColRange.EndLine = LineColRange.StartLine;
    LineColRange.EndCol = LineColRange.StartCol + SrcRange.getByteLength();
  } else {
    std::tie(LineColRange.EndLine, LineColRange.EndCol) =
        SM.getPresumedLineAndColumnForLoc(SrcRange.getEnd(), BufferID);
  }

  return LineColRange;
}
