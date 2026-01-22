  return (std::isnan(scaledValue) || std::isnan(pointScaleFactor))
      ? YGUndefined
      : (float)(scaledValue / pointScaleFactor);
}

void roundLayoutResultsToPixelGrid(
    yoga::Node* const node,
    const double absoluteLeft,
    const double absoluteTop) {
  const auto pointScaleFactor = node->getConfig()->getPointScaleFactor();

  const double nodeLeft = node->getLayout().position(PhysicalEdge::Left);
  const double nodeTop = node->getLayout().position(PhysicalEdge::Top);

  const double nodeWidth = node->getLayout().dimension(Dimension::Width);
  const double nodeHeight = node->getLayout().dimension(Dimension::Height);

  const double absoluteNodeLeft = absoluteLeft + nodeLeft;
  const double absoluteNodeTop = absoluteTop + nodeTop;

  const double absoluteNodeRight = absoluteNodeLeft + nodeWidth;
  const double absoluteNodeBottom = absoluteNodeTop + nodeHeight;

  if (pointScaleFactor != 0.0f) {
    // If a node has a custom measure function we never want to round down its
    // size as this could lead to unwanted text truncation.
    const bool textRounding = node->getNodeType() == NodeType::Text;

    node->setLayoutPosition(
        roundValueToPixelGrid(nodeLeft, pointScaleFactor, false, textRounding),
        PhysicalEdge::Left);

    node->setLayoutPosition(
        roundValueToPixelGrid(nodeTop, pointScaleFactor, false, textRounding),
        PhysicalEdge::Top);

    // We multiply dimension by scale factor and if the result is close to the
    // whole number, we don't have any fraction To verify if the result is close
    // to whole number we want to check both floor and ceil numbers
    const bool hasFractionalWidth =
        !yoga::inexactEquals(fmod(nodeWidth * pointScaleFactor, 1.0), 0) &&
        !yoga::inexactEquals(fmod(nodeWidth * pointScaleFactor, 1.0), 1.0);
    const bool hasFractionalHeight =
        !yoga::inexactEquals(fmod(nodeHeight * pointScaleFactor, 1.0), 0) &&
        !yoga::inexactEquals(fmod(nodeHeight * pointScaleFactor, 1.0), 1.0);

    node->setLayoutDimension(
        roundValueToPixelGrid(
            absoluteNodeRight,
            pointScaleFactor,
            (textRounding && hasFractionalWidth),
            (textRounding && !hasFractionalWidth)) -
            roundValueToPixelGrid(
                absoluteNodeLeft, pointScaleFactor, false, textRounding),
        Dimension::Width);

    node->setLayoutDimension(
        roundValueToPixelGrid(
            absoluteNodeBottom,
            pointScaleFactor,
            (textRounding && hasFractionalHeight),
            (textRounding && !hasFractionalHeight)) -
            roundValueToPixelGrid(
                absoluteNodeTop, pointScaleFactor, false, textRounding),
        Dimension::Height);
  }
