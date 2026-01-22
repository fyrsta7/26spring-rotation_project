    if (valid() && value == value_) {
      if (assumeDistinct == true) {
        return true;
      }

      // We might be in the middle of a run of equal values, reposition by
      // iterating backwards to its first element.
      auto valueLower = Instructions::bzhi(value_, numLowerBits_);
      while (!upper_.isAtBeginningOfRun() &&
             readLowerPart(position() - 1) == valueLower) {
