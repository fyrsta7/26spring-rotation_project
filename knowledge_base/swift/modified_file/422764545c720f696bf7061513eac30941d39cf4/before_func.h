  size_t hash() {
    size_t H = 0x56ba80d1 ^ length ;
    for (unsigned i = 0; i < length; i++) {
      H = (H >> 10) | (H << ((sizeof(size_t) * 8) - 10));
      H ^= ((size_t)args[i]) ^ ((size_t)args[i] >> 19);
    }
    return H * 0x27d4eb2d;
  }
