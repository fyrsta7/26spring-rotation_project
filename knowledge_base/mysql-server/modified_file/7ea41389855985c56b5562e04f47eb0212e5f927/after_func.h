
  bool Init() override;
  int Read() override;
  void SetNullRowFlag(bool is_null_row) override {
    m_iterator.SetNullRowFlag(is_null_row);
  }
  void UnlockRow() override { m_iterator.UnlockRow(); }
  std::vector<Child> children() const override { return m_iterator.children(); }
  std::vector<std::string> DebugString() const override {
    return m_iterator.DebugString();
  }
  void StartPSIBatchMode() override { m_iterator.StartPSIBatchMode(); }
  void EndPSIBatchModeIfStarted() override {
    m_iterator.EndPSIBatchModeIfStarted();
  }

  std::string TimingString() const override;

  RowIterator *real_iterator() override { return &m_iterator; }
