
  in.skipWhitespace();
  return *in == '[' ? parseArray(in) :
         *in == '{' ? parseObject(in) :
         *in == '\"' ? parseString(in) :
         (*in == '-' || (*in >= '0' && *in <= '9')) ? parseNumber(in) :
         in.consume("true") ? true :
         in.consume("false") ? false :
         in.consume("null") ? nullptr :
         in.consume("Infinity") ?
          (in.getOpts().parse_numbers_as_strings ? (dynamic)"Infinity" :
            (dynamic)std::numeric_limits<double>::infinity()) :
         in.consume("NaN") ?
           (in.getOpts().parse_numbers_as_strings ? (dynamic)"NaN" :
             (dynamic)std::numeric_limits<double>::quiet_NaN()) :
         in.error("expected json value");
}

}

//////////////////////////////////////////////////////////////////////

std::string serialize(dynamic const& dyn, serialization_opts const& opts) {
  std::string ret;
  unsigned indentLevel = 0;
  Printer p(ret, opts.pretty_formatting ? &indentLevel : nullptr, &opts);
  p(dyn);
  return ret;
}

// Escape a string so that it is legal to print it in JSON text.
void escapeString(
    StringPiece input,
    std::string& out,
    const serialization_opts& opts) {
  auto hexDigit = [] (int c) -> char {
    return c < 10 ? c + '0' : c - 10 + 'a';
  };

  out.reserve(out.size() + input.size() + 2);
  out.push_back('\"');

  auto* p = reinterpret_cast<const unsigned char*>(input.begin());
  auto* q = reinterpret_cast<const unsigned char*>(input.begin());
  auto* e = reinterpret_cast<const unsigned char*>(input.end());

  while (p < e) {
    // Since non-ascii encoding inherently does utf8 validation
    // we explicitly validate utf8 only if non-ascii encoding is disabled.
    if ((opts.validate_utf8 || opts.skip_invalid_utf8)
        && !opts.encode_non_ascii) {
      // to achieve better spatial and temporal coherence
      // we do utf8 validation progressively along with the
      // string-escaping instead of two separate passes

      // as the encoding progresses, q will stay at or ahead of p
      CHECK(q >= p);

      // as p catches up with q, move q forward
      if (q == p) {
        // calling utf8_decode has the side effect of
        // checking that utf8 encodings are valid
        char32_t v = decodeUtf8(q, e, opts.skip_invalid_utf8);
        if (opts.skip_invalid_utf8 && v == U'\ufffd') {
          out.append(u8"\ufffd");
          p = q;
          continue;
        }
      }
    }
    if (opts.encode_non_ascii && (*p & 0x80)) {
      // note that this if condition captures utf8 chars
      // with value > 127, so size > 1 byte
