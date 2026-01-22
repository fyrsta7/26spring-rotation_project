void Editor::drawOneSpriteUnclippedRect(ui::Graphics* g, const gfx::Rect& spriteRectToDraw, int dx, int dy)
{
  // Clip from sprite and apply zoom
  gfx::Rect rc = m_sprite->bounds().createIntersection(spriteRectToDraw);
  rc = m_proj.apply(rc);

  gfx::Rect dest(dx + m_padding.x + rc.x,
                 dy + m_padding.y + rc.y, 0, 0);

  // Clip from graphics/screen
  const gfx::Rect& clip = g->getClipBounds();
  if (dest.x < clip.x) {
    rc.x += clip.x - dest.x;
    rc.w -= clip.x - dest.x;
    dest.x = clip.x;
  }
  if (dest.y < clip.y) {
    rc.y += clip.y - dest.y;
    rc.h -= clip.y - dest.y;
    dest.y = clip.y;
  }
  if (dest.x+rc.w > clip.x+clip.w) {
    rc.w = clip.x+clip.w-dest.x;
  }
  if (dest.y+rc.h > clip.y+clip.h) {
    rc.h = clip.y+clip.h-dest.y;
  }

  if (rc.isEmpty())
    return;

  // Bounds of pixels from the sprite canvas that will be exposed in
  // this render cycle.
  gfx::Rect expose = m_proj.remove(rc);

  // If the zoom level is less than 100%, we add extra pixels to
  // the exposed area. Those pixels could be shown in the
  // rendering process depending on each cel position.
  // E.g. when we are drawing in a cel with position < (0,0)
  if (m_proj.scaleX() < 1.0)
    expose.enlargeXW(int(1./m_proj.scaleX()));
  // If the zoom level is more than %100 we add an extra pixel to
  // expose just in case the zoom requires to display it.  Note:
  // this is really necessary to avoid showing invalid destination
  // areas in ToolLoopImpl.
  else if (m_proj.scaleX() > 1.0)
    expose.enlargeXW(1);

  if (m_proj.scaleY() < 1.0)
    expose.enlargeYH(int(1./m_proj.scaleY()));
  else if (m_proj.scaleY() > 1.0)
    expose.enlargeYH(1);

  expose &= m_sprite->bounds();

  const int maxw = std::max(0, m_sprite->width()-expose.x);
  const int maxh = std::max(0, m_sprite->height()-expose.y);
  expose.w = MID(0, expose.w, maxw);
  expose.h = MID(0, expose.h, maxh);
  if (expose.isEmpty())
    return;

  // rc2 is the rectangle used to create a temporal rendered image of the sprite
  const bool newEngine =
    (Preferences::instance().experimental.newRenderEngine()
     // Reference layers + zoom > 100% need the old render engine for
     // sub-pixel rendering.
     && (!m_sprite->hasVisibleReferenceLayers()
         || (m_proj.scaleX() <= 1.0
             && m_proj.scaleY() <= 1.0)));
  gfx::Rect rc2;
  if (newEngine) {
    rc2 = expose;               // New engine, exposed rectangle (without zoom)
    dest.x = dx + m_padding.x + m_proj.applyX(rc2.x);
    dest.y = dy + m_padding.y + m_proj.applyY(rc2.y);
    dest.w = m_proj.applyX(rc2.w);
    dest.h = m_proj.applyY(rc2.h);
  }
  else {
    rc2 = rc;                   // Old engine, same rectangle with zoom
    dest.w = rc.w;
    dest.h = rc.h;
  }

  std::unique_ptr<Image> rendered(nullptr);
  try {
    // Generate a "expose sprite pixels" notification. This is used by
    // tool managers that need to validate this region (copy pixels from
    // the original cel) before it can be used by the RenderEngine.
    m_document->notifyExposeSpritePixels(m_sprite, gfx::Region(expose));

    // Create a temporary RGB bitmap to draw all to it
    rendered.reset(Image::create(IMAGE_RGB, rc2.w, rc2.h,
                                 m_renderEngine->getRenderImageBuffer()));

    m_renderEngine->setNewBlendMethod(Preferences::instance().experimental.newBlend());
    m_renderEngine->setRefLayersVisiblity(true);
    m_renderEngine->setSelectedLayer(m_layer);
    if (m_flags & Editor::kUseNonactiveLayersOpacityWhenEnabled)
      m_renderEngine->setNonactiveLayersOpacity(Preferences::instance().experimental.nonactiveLayersOpacity());
    else
      m_renderEngine->setNonactiveLayersOpacity(255);
    m_renderEngine->setProjection(
      newEngine ? render::Projection(): m_proj);
    m_renderEngine->setupBackground(m_document, rendered->pixelFormat());
    m_renderEngine->disableOnionskin();

    if ((m_flags & kShowOnionskin) == kShowOnionskin) {
      if (m_docPref.onionskin.active()) {
        OnionskinOptions opts(
          (m_docPref.onionskin.type() == app::gen::OnionskinType::MERGE ?
           render::OnionskinType::MERGE:
           (m_docPref.onionskin.type() == app::gen::OnionskinType::RED_BLUE_TINT ?
            render::OnionskinType::RED_BLUE_TINT:
            render::OnionskinType::NONE)));

        opts.position(m_docPref.onionskin.position());
        opts.prevFrames(m_docPref.onionskin.prevFrames());
        opts.nextFrames(m_docPref.onionskin.nextFrames());
        opts.opacityBase(m_docPref.onionskin.opacityBase());
        opts.opacityStep(m_docPref.onionskin.opacityStep());
        opts.layer(m_docPref.onionskin.currentLayer() ? m_layer: nullptr);

        FrameTag* tag = nullptr;
        if (m_docPref.onionskin.loopTag())
          tag = m_sprite->frameTags().innerTag(m_frame);
        opts.loopTag(tag);

        m_renderEngine->setOnionskin(opts);
      }
    }

    ExtraCelRef extraCel = m_document->extraCel();
    if (extraCel && extraCel->type() != render::ExtraType::NONE) {
      m_renderEngine->setExtraImage(
        extraCel->type(),
        extraCel->cel(),
        extraCel->image(),
        extraCel->blendMode(),
        m_layer, m_frame);
    }

    m_renderEngine->renderSprite(
      rendered.get(), m_sprite, m_frame, gfx::Clip(0, 0, rc2));

    m_renderEngine->removeExtraImage();
  }
  catch (const std::exception& e) {
    Console::showException(e);
  }

  if (rendered) {
    // Convert the render to a os::Surface
    static os::Surface* tmp = nullptr; // TODO move this to other centralized place

    if (!tmp ||
        tmp->width() < rc2.w ||
        tmp->height() < rc2.h ||
        tmp->colorSpace() != m_document->osColorSpace()) {
      const int maxw = std::max(rc2.w, tmp ? tmp->width(): 0);
      const int maxh = std::max(rc2.h, tmp ? tmp->height(): 0);
      if (tmp)
        tmp->dispose();

      tmp = os::instance()->createSurface(
        maxw, maxh, m_document->osColorSpace());
    }

    if (tmp->nativeHandle()) {
      convert_image_to_surface(rendered.get(), m_sprite->palette(m_frame),
                               tmp, 0, 0, 0, 0, rc2.w, rc2.h);

      if (newEngine) {
        g->drawSurface(tmp, gfx::Rect(0, 0, rc2.w, rc2.h), dest);
      }
      else {
        g->blit(tmp, 0, 0, dest.x, dest.y, dest.w, dest.h);
      }
      m_brushPreview.invalidateRegion(gfx::Region(dest));
    }
  }

  // Draw grids
  {
    gfx::Rect enclosingRect(
      m_padding.x + dx,
      m_padding.y + dy,
      m_proj.applyX(m_sprite->width()),
      m_proj.applyY(m_sprite->height()));

    IntersectClip clip(g, dest);
    if (clip) {
      // Draw the pixel grid
      if ((m_proj.zoom().scale() > 2.0) && m_docPref.show.pixelGrid()) {
        int alpha = m_docPref.pixelGrid.opacity();

        if (m_docPref.pixelGrid.autoOpacity()) {
          alpha = int(alpha * (m_proj.zoom().scale()-2.) / (16.-2.));
          alpha = MID(0, alpha, 255);
        }

        drawGrid(g, enclosingRect, Rect(0, 0, 1, 1),
                 m_docPref.pixelGrid.color(), alpha);
      }

      // Draw the grid
      if (m_docPref.show.grid()) {
        gfx::Rect gridrc = m_docPref.grid.bounds();
        if (m_proj.applyX(gridrc.w) > 2 &&
            m_proj.applyY(gridrc.h) > 2) {
          int alpha = m_docPref.grid.opacity();

          if (m_docPref.grid.autoOpacity()) {
            double len = (m_proj.applyX(gridrc.w) +
                          m_proj.applyY(gridrc.h)) / 2.;
            alpha = int(alpha * len / 32.);
            alpha = MID(0, alpha, 255);
          }

          if (alpha > 8)
            drawGrid(g, enclosingRect, m_docPref.grid.bounds(),
                     m_docPref.grid.color(), alpha);
        }
      }
    }
  }
}
