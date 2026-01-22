  }
  return false;
}

inline uint char_val(char X) {
  return (uint)(X >= '0' && X <= '9'
                    ? X - '0'
                    : X >= 'A' && X <= 'Z' ? X - 'A' + 10 : X - 'a' + 10);
}

Item_hex_string::Item_hex_string() { hex_string_init("", 0); }
