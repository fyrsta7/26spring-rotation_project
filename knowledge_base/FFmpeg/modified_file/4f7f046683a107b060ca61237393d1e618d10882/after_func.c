
    for (;;) {
        draw_dot(s, x0, y0);

        if (x0 == x1 && y0 == y1)
            break;

        e2 = err;

        if (e2 >-dx) {
            err -= dy;
            x0 += sx;
        }

        if (e2 < dy) {
            err += dx;
            y0 += sy;
        }
    }
}

static void fade(AudioVectorScopeContext *s)
{
    const int linesize = s->outpicref->linesize[0];
