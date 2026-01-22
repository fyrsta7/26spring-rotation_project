void String::TryFlatten() {
  // We don't need to flatten strings that are already flat.  Since this code
  // is inlined, it can be helpful in the flat case to not call out to Flatten.
  StringRepresentationTag str_type = representation_tag();
  if (str_type != kSeqStringTag && str_type != kExternalStringTag) {
    Flatten();
  }
}
