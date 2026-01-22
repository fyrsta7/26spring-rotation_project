void OctreeMap::feedWithImage(const Image* image,
                              const bool withAlpha,
                              const color_t maskColor,
                              const int levelDeep)
{
  ASSERT(image);
  ASSERT(image->pixelFormat() == IMAGE_RGB || image->pixelFormat() == IMAGE_GRAYSCALE);
  color_t forceFullOpacity;
  const bool imageIsRGBA = image->pixelFormat() == IMAGE_RGB;

  auto add_color_to_octree =
    [this, &forceFullOpacity, &levelDeep, &imageIsRGBA](color_t color) {
      const int alpha = (imageIsRGBA ? rgba_geta(color) : graya_geta(color));
      if (alpha >= MIN_ALPHA_THRESHOLD) { // Colors which alpha is less than
                                          // MIN_ALPHA_THRESHOLD will not registered
        color |= forceFullOpacity;
        color = (imageIsRGBA ? color : rgba(graya_getv(color),
                                            graya_getv(color),
                                            graya_getv(color),
                                            alpha));
        addColor(color, levelDeep);
      }
    };

  switch (image->pixelFormat()) {
    case IMAGE_RGB: {
      forceFullOpacity = (withAlpha) ? 0 : rgba_a_mask;
      doc::for_each_pixel<RgbTraits>(image, add_color_to_octree);
      break;
    }
    case IMAGE_GRAYSCALE: {
      forceFullOpacity = (withAlpha) ? 0 : graya_a_mask;
      doc::for_each_pixel<GrayscaleTraits>(image, add_color_to_octree);
      break;
    }
  }
  m_maskColor = maskColor;
}
