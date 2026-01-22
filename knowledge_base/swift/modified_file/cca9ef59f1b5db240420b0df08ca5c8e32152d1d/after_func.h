  bool operator==(const ImmutablePointerSet<T> &P) const {
    // If this and P have different sizes, we can not be equivalent.
    if (size() != P.size())
      return false;

    // Ok, we now know that both have the same size. If one is empty, the other
    // must be as well, implying equality.
    if (empty())
      return true;

    // Ok, both sets are not empty and the same number of elements. Compare
    // element wise.
    return std::equal(begin(), end(), P.begin());
  }
