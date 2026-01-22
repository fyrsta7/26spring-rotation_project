inline bool LoadDataFromFile(const STRING& filename,
                             GenericVector<char>* data) {
  FILE* fp = fopen(filename.string(), "rb");
  if (fp == NULL) return false;
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  data->init_to_size(static_cast<int>(size), 0);
  bool result = fread(&(*data)[0], 1, size, fp) == size;
  fclose(fp);
  return result;
}
