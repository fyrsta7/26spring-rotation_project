  void setFinalStep(ToolLoop* loop, bool state) override {
    m_modify_selection = state;

    if (state) {
      m_maxBounds = loop->getMask()->bounds();

      m_mask.copyFrom(loop->getMask());
      m_mask.freeze();
      m_mask.reserve(loop->sprite()->bounds());
    }
    else {
      int modifiers = int(loop->getModifiers());
      if ((modifiers & int(ToolLoopModifiers::kIntersectSelection)) != 0) {
        m_mask.intersect(m_intersectMask);
      }

      // We can intersect the used bounds in inkHline() calls to
      // reduce the shrink computation.
      m_mask.intersect(m_maxBounds);

      m_mask.unfreeze();

      loop->setMask(&m_mask);
      loop->getDocument()->setTransformation(
        Transformation(RectF(m_mask.bounds())));

      m_mask.clear();
    }
  }
