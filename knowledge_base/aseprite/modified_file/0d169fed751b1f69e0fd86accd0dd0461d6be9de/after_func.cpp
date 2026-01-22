void drawTextBox(Graphics* g, Widget* widget,
                 int* w, int* h, gfx::Color bg, gfx::Color fg)
{
  View* view = View::getView(widget);
  char* text = const_cast<char*>(widget->text().c_str());
  char* beg, *end;
  int x1, y1, x2, y2;
  int x, y, chr, len;
  gfx::Point scroll;
  int textheight = widget->textHeight();
  she::Font* font = widget->font();
  char *beg_end, *old_end;
  int width;
  gfx::Rect vp;

  if (view) {
    vp = view->viewportBounds().offset(-widget->bounds().origin());
    scroll = view->viewScroll();
  }
  else {
    vp = widget->clientBounds();
    vp.w -= widget->border().width();
    vp.h -= widget->border().height();
    scroll.x = scroll.y = 0;
  }
  x1 = widget->clientBounds().x;
  y1 = widget->clientBounds().y;
  x2 = vp.x + vp.w;
  y2 = vp.y + vp.h;

  // Fill background
  if (g)
    g->fillRect(bg, vp);

  chr = 0;

  // Without word-wrap
  if (!(widget->align() & WORDWRAP)) {
    width = widget->clientBounds().w;
  }
  // With word-wrap
  else {
    if (w) {
      width = *w;
      *w = 0;
    }
    else {
      // TODO modificable option? I don't think so, this is very internal stuff
#if 0
      // Shows more information in x-scroll 0
      width = vp.w;
#else
      // Make good use of the complete text-box
      if (view) {
        gfx::Size maxSize = view->getScrollableSize();
        width = MAX(vp.w, maxSize.w);
      }
      else {
        width = vp.w;
      }
#endif
    }
  }

  // Draw line-by-line
  y = y1;
  for (beg=end=text; end; ) {
    x = x1;

    // Without word-wrap
    if (!(widget->align() & WORDWRAP)) {
      end = std::strchr(beg, '\n');
      if (end) {
        chr = *end;
        *end = 0;
      }
    }
    // With word-wrap
    else {
      old_end = NULL;
      for (beg_end=beg;;) {
        end = std::strpbrk(beg_end, " \n");
        if (end) {
          chr = *end;
          *end = 0;
        }

        // To here we can print
        if ((old_end) && (x+font->textLength(beg) > x1-scroll.x+width)) {
          if (end)
            *end = chr;

          end = old_end;
          chr = *end;
          *end = 0;
          break;
        }
        // We can print one word more
        else if (end) {
          // Force break
          if (chr == '\n')
            break;

          *end = chr;
          beg_end = end+1;
        }
        // We are in the end of text
        else
          break;

        old_end = end;
      }
    }

    len = font->textLength(beg);

    // Render the text
    if (g) {
      int xout;

      if (widget->align() & CENTER)
        xout = x + width/2 - len/2;
      else if (widget->align() & RIGHT)
        xout = x + width - len;
      else                      // Left align
        xout = x;

      g->drawUIString(beg, fg, gfx::ColorNone, gfx::Point(xout, y));
    }

    if (w)
      *w = MAX(*w, len);

    y += textheight;

    if (end) {
      *end = chr;
      beg = end+1;
    }
  }

  if (h)
    *h = (y - y1 + scroll.y);

  if (w) *w += widget->border().width();
  if (h) *h += widget->border().height();
}
