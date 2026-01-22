void PixelsMovement::drawParallelogram(
  doc::Image* dst, const doc::Image* src, const doc::Mask* mask,
  const Transformation::Corners& corners,
  const gfx::Point& leftTop)
{
  tools::RotationAlgorithm rotAlgo = Preferences::instance().selection.rotationAlgorithm();

  // When the scale isn't modified and we have no rotation or a
  // right/straight-angle, we should use the fast rotation algorithm,
  // as it's pixel-perfect match with the original selection when just
  // a translation is applied.
  double angle = 180.0*m_currentData.angle()/PI;
  if (std::fabs(std::fmod(std::fabs(angle), 90.0)) < 0.01 ||
      std::fabs(std::fmod(std::fabs(angle), 90.0)-90.0) < 0.01) {
    rotAlgo = tools::RotationAlgorithm::FAST;
  }

retry:;      // In case that we don't have enough memory for RotSprite
             // we can try with the fast algorithm anyway.

  switch (rotAlgo) {

    case tools::RotationAlgorithm::FAST:
      doc::algorithm::parallelogram(
        dst, src, (mask ? mask->bitmap(): nullptr),
        int(corners.leftTop().x-leftTop.x),
        int(corners.leftTop().y-leftTop.y),
        int(corners.rightTop().x-leftTop.x),
        int(corners.rightTop().y-leftTop.y),
        int(corners.rightBottom().x-leftTop.x),
        int(corners.rightBottom().y-leftTop.y),
        int(corners.leftBottom().x-leftTop.x),
        int(corners.leftBottom().y-leftTop.y));
      break;

    case tools::RotationAlgorithm::ROTSPRITE:
      try {
        doc::algorithm::rotsprite_image(
          dst, src, (mask ? mask->bitmap(): nullptr),
          int(corners.leftTop().x-leftTop.x),
          int(corners.leftTop().y-leftTop.y),
          int(corners.rightTop().x-leftTop.x),
          int(corners.rightTop().y-leftTop.y),
          int(corners.rightBottom().x-leftTop.x),
          int(corners.rightBottom().y-leftTop.y),
          int(corners.leftBottom().x-leftTop.x),
          int(corners.leftBottom().y-leftTop.y));
      }
      catch (const std::bad_alloc&) {
        StatusBar::instance()->showTip(1000,
          "Not enough memory for RotSprite");

        rotAlgo = tools::RotationAlgorithm::FAST;
        goto retry;
      }
      break;

  }
}
