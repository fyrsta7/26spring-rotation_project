    }
  }

  if (!mainFileName.empty()) {
    locationInfo.hasMainFile = true;
    locationInfo.mainFile = Path(compilationDirectory, "", mainFileName);
  }

  if (!foundLineOffset) {
    return false;
  }

  folly::StringPiece lineSection(line_);
  lineSection.advance(lineOffset);
  LineNumberVM lineVM(lineSection, compilationDirectory);

  // Execute line number VM program to find file and line
  locationInfo.hasFileAndLine =
      lineVM.findAddress(address, locationInfo.file, locationInfo.line);
  return locationInfo.hasFileAndLine;
}

bool Dwarf::findAddress(uintptr_t address, LocationInfo& locationInfo) const {
  locationInfo = LocationInfo();

  if (!elf_) { // no file
    return false;
  }

  if (!aranges_.empty()) {
