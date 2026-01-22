}

/* static */ bool ShapeUtil::EqualIgnoringFpPrecision(const Shape& lhs,
                                                      const Shape& rhs) {
  bool equal = Shape::Equal().IgnoreFpPrecision()(lhs, rhs);
  if (!equal && VLOG_IS_ON(3)) {
    VLOG(3) << "ShapeUtil::EqualIgnoringFpPrecision differ: lhs = "
            << lhs.ShortDebugString() << ", rhs = " << rhs.ShortDebugString();
  }

  return equal;
}
