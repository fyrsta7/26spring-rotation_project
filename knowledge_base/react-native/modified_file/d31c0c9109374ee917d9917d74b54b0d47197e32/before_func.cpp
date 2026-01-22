    return lhs == rhs;
  }
  return std::string(lhs) == std::string(rhs);
}

bool operator==(RawPropsKey const &lhs, RawPropsKey const &rhs) noexcept {
