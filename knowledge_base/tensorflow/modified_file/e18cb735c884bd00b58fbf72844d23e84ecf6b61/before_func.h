    return *(new (&storage_.back()) T(std::forward<T>(value)));
  }

