void Editor::drawSpriteClipped(const gfx::Region& updateRegion)
{
  Region screenRegion;
  getDrawableRegion(screenRegion, kCutTopWindows);

  ScreenGraphics screenGraphics;
  GraphicsPtr editorGraphics = getGraphics(clientBounds());

  for (const Rect& updateRect : updateRegion) {
    Rect spriteRectOnScreen = editorToScreen(updateRect);

    for (const Rect& screenRect : screenRegion) {
      IntersectClip clip(&screenGraphics,
                         screenRect & spriteRectOnScreen);
      if (clip)
        drawSpriteUnclippedRect(editorGraphics.get(), updateRect);
    }
  }
}
