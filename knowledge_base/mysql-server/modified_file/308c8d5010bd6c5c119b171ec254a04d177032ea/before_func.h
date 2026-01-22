
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
