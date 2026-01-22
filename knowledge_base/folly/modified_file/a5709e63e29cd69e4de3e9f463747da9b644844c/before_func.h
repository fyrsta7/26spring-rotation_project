private:

  // std::vector implements a similar function with a different growth
  //  strategy: empty() ? 1 : capacity() * 2.
  //
  // fbvector grows differently on two counts:
  //
  // (1) initial size
  //     Instead of grwoing to size 1 from empty, and fbvector allocates at
  //     least 64 bytes. You may still use reserve to reserve a lesser amount
