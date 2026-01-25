  // Constructor with a serialized string object
  explicit WriteBatch(const std::string& rep) : save_points_(nullptr), rep_(rep) {}