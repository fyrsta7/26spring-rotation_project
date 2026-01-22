int Map::NextFreePropertyIndex() const {
  int number_of_own_descriptors = NumberOfOwnDescriptors();
  DescriptorArray descs = instance_descriptors();
  // Search properties backwards to find the last field.
  for (int i = number_of_own_descriptors - 1; i >= 0; --i) {
    PropertyDetails details = descs.GetDetails(i);
    if (details.location() == kField) {
      return details.field_index() + details.field_width_in_words();
    }
  }
  return 0;
}
