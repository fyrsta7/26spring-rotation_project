void Widget::setBoundsQuietly(const gfx::Rect& rc)
{
  if (m_bounds != rc) {
    m_bounds = rc;

    // Remove all paint messages for this widget.
    if (Manager* manager = this->manager())
      manager->removeMessagesFor(this, kPaintMessage);
  }

  invalidate();
}
