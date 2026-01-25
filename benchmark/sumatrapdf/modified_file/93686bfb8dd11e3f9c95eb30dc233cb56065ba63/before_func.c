static void
fz_insert_gel_raw(fz_gel *gel, int x0, int y0, int x1, int y1)
{
	fz_edge *edge;
	int dx, dy;
	int winding;
	int width;
	int tmp;

	if (y0 == y1)
		return;

	if (y0 > y1) {
		winding = -1;
		tmp = x0; x0 = x1; x1 = tmp;
		tmp = y0; y0 = y1; y1 = tmp;
	}
	else
		winding = 1;

	if (x0 < gel->bbox.x0) gel->bbox.x0 = x0;
	if (x0 > gel->bbox.x1) gel->bbox.x1 = x0;
	if (x1 < gel->bbox.x0) gel->bbox.x0 = x1;
	if (x1 > gel->bbox.x1) gel->bbox.x1 = x1;

	if (y0 < gel->bbox.y0) gel->bbox.y0 = y0;
	if (y1 > gel->bbox.y1) gel->bbox.y1 = y1;

	if (gel->len + 1 == gel->cap) {
		int new_cap = gel->cap + 512;
		gel->edges = fz_resize_array(gel->ctx, gel->edges, new_cap, sizeof(fz_edge));
		gel->cap = new_cap;
	}

	edge = &gel->edges[gel->len++];

	dy = y1 - y0;
	dx = x1 - x0;
	width = fz_absi(dx);

	edge->xdir = dx > 0 ? 1 : -1;
	edge->ydir = winding;
	edge->x = x0;
	edge->y = y0;
	edge->h = dy;
	edge->adj_down = dy;

	/* initial error term going l->r and r->l */
	if (dx >= 0)
		edge->e = 0;
	else
		edge->e = -dy + 1;

	/* y-major edge */
	if (dy >= width) {
		edge->xmove = 0;
		edge->adj_up = width;
	}

	/* x-major edge */
	else {
		edge->xmove = (width / dy) * edge->xdir;
		edge->adj_up = width % dy;
	}
}