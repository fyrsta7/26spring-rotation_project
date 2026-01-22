int Map::NextFreePropertyIndex() const {
  int free_index = 0;
  int number_of_own_descriptors = NumberOfOwnDescriptors();
  DescriptorArray descs = instance_descriptors();
  for (int i = 0; i < number_of_own_descriptors; i++) {
    PropertyDetails details = descs.GetDetails(i);
    if (details.location() == kField) {
      int candidate = details.field_index() + details.field_width_in_words();
      if (candidate > free_index) free_index = candidate;
    }
  }
  return free_index;
}
