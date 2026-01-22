  ~ScopedVariable() {
    // Explicitly mark the variable as invalid to avoid the creation of
    // unnecessary loop phis.
    assembler_.SetVariable(*this, OpIndex::Invalid());
  }
