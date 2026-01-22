    char termChar = '\0',
    size_t maxLength = std::numeric_limits<size_t>::max()) {
    std::string str;

    for (;;) {
      const uint8_t* buf = data();
      size_t buflen = length();

      size_t i = 0;
