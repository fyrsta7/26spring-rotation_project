    return *(new (&storage_.back()) T(std::forward<T>(value)));
  }

  template <typename T = std::tuple_element_t<0, std::tuple<Ts...>>,
            typename... Args>
  T& emplace_back(Args... args) {
