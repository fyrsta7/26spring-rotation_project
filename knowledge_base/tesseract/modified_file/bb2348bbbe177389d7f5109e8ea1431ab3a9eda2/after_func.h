inline bool LoadDataFromFile(const STRING& filename,
                             GenericVector<char>* data) {
  bool result = false;
  FILE* fp = fopen(filename.string(), "rb");
  if (fp != NULL) {
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size > 0) {
      data->resize_no_init(size);
      result = fread(&(*data)[0], 1, size, fp) == size;
    }
    fclose(fp);
  }
  return result;
}
