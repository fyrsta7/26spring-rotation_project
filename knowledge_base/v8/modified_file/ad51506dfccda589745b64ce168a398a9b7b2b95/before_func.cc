  CHECK_EQ(no_info, info.GetSourceLineNumber(21));
  CHECK_EQ(no_info, info.GetSourceLineNumber(100));
  CHECK_EQ(no_info, info.GetSourceLineNumber(std::numeric_limits<int>::max()));

  info.SetPosition(10, 1);
  info.SetPosition(20, 2);

  // The only valid return values are 1 or 2 - every pc maps to a line number.
  CHECK_EQ(1, info.GetSourceLineNumber(std::numeric_limits<int>::min()));
  CHECK_EQ(1, info.GetSourceLineNumber(0));
  CHECK_EQ(1, info.GetSourceLineNumber(1));
  CHECK_EQ(1, info.GetSourceLineNumber(9));
  CHECK_EQ(1, info.GetSourceLineNumber(10));
  CHECK_EQ(1, info.GetSourceLineNumber(11));
  CHECK_EQ(1, info.GetSourceLineNumber(19));
  CHECK_EQ(2, info.GetSourceLineNumber(20));
  CHECK_EQ(2, info.GetSourceLineNumber(21));
  CHECK_EQ(2, info.GetSourceLineNumber(100));
  CHECK_EQ(2, info.GetSourceLineNumber(std::numeric_limits<int>::max()));

  // Test SetPosition behavior.
  info.SetPosition(25, 3);
  CHECK_EQ(2, info.GetSourceLineNumber(21));
  CHECK_EQ(3, info.GetSourceLineNumber(100));
  CHECK_EQ(3, info.GetSourceLineNumber(std::numeric_limits<int>::max()));
}
