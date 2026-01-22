                   if (pwidth != nullptr) *pwidth = width;
                   if (pheight != nullptr) *pheight = height;
                   if (pcomponents != nullptr) *pcomponents = components;
                   buffer = new uint8[height * width * components];
                   return buffer;
                 });
  if (!result) delete[] buffer;
  return result;
}

// ----------------------------------------------------------------------------
// Computes image information from jpeg header.
// Returns true on success; false on failure.
bool GetImageInfo(const void* srcdata, int datasize, int* width, int* height,
                  int* components) {
  // Init in case of failure
  if (width) *width = 0;
  if (height) *height = 0;
  if (components) *components = 0;

  // If empty image, return
  if (datasize == 0 || srcdata == nullptr) return false;

  // Initialize libjpeg structures to have a memory source
  // Modify the usual jpeg error manager to catch fatal errors.
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  jmp_buf jpeg_jmpbuf;
  cinfo.err = jpeg_std_error(&jerr);
  cinfo.client_data = &jpeg_jmpbuf;
  jerr.error_exit = CatchError;
  if (setjmp(jpeg_jmpbuf)) {
    return false;
  }

  // set up, read header, set image parameters, save size
