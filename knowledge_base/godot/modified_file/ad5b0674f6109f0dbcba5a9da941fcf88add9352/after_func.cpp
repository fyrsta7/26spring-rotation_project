inline void draw_ring(Vector<Vector2> &verts, Vector<int> &indices, Vector<Color> &colors, const Rect2 style_rect, const int corner_radius[4],
		const Rect2 ring_rect, const int border_width[4], const Color inner_color[4], const Color outer_color[4], const int corner_detail) {

	int vert_offset = verts.size();
	if (!vert_offset) {
		vert_offset = 0;
	}
	int adapted_corner_detail = (corner_radius[0] == 0 && corner_radius[1] == 0 && corner_radius[2] == 0 && corner_radius[3] == 0) ? 1 : corner_detail;
	int rings = (border_width[0] == 0 && border_width[1] == 0 && border_width[2] == 0 && border_width[3] == 0) ? 1 : 2;
	rings = 2;

	int ring_corner_radius[4];
	set_inner_corner_radius(style_rect, ring_rect, corner_radius, ring_corner_radius);

	//corner radius center points
	Vector<Point2> outer_points;
	outer_points.push_back(ring_rect.position + Vector2(ring_corner_radius[0], ring_corner_radius[0])); //tl
	outer_points.push_back(Point2(ring_rect.position.x + ring_rect.size.x - ring_corner_radius[1], ring_rect.position.y + ring_corner_radius[1])); //tr
	outer_points.push_back(ring_rect.position + ring_rect.size - Vector2(ring_corner_radius[2], ring_corner_radius[2])); //br
	outer_points.push_back(Point2(ring_rect.position.x + ring_corner_radius[3], ring_rect.position.y + ring_rect.size.y - ring_corner_radius[3])); //bl

	Rect2 inner_rect;
	inner_rect = ring_rect.grow_individual(-border_width[MARGIN_LEFT], -border_width[MARGIN_TOP], -border_width[MARGIN_RIGHT], -border_width[MARGIN_BOTTOM]);
	int inner_corner_radius[4];

	Vector<Point2> inner_points;
	set_inner_corner_radius(style_rect, inner_rect, corner_radius, inner_corner_radius);
	inner_points.push_back(inner_rect.position + Vector2(inner_corner_radius[0], inner_corner_radius[0])); //tl
	inner_points.push_back(Point2(inner_rect.position.x + inner_rect.size.x - inner_corner_radius[1], inner_rect.position.y + inner_corner_radius[1])); //tr
	inner_points.push_back(inner_rect.position + inner_rect.size - Vector2(inner_corner_radius[2], inner_corner_radius[2])); //br
	inner_points.push_back(Point2(inner_rect.position.x + inner_corner_radius[3], inner_rect.position.y + inner_rect.size.y - inner_corner_radius[3])); //bl

	//calculate the vert array
	for (int corner_index = 0; corner_index < 4; corner_index++) {
		for (int detail = 0; detail <= adapted_corner_detail; detail++) {
			for (int inner_outer = (2 - rings); inner_outer < 2; inner_outer++) {
				float radius;
				Color color;
				Point2 corner_point;
				if (inner_outer == 0) {
					radius = inner_corner_radius[corner_index];
					color = *inner_color;
					corner_point = inner_points[corner_index];
				} else {
					radius = ring_corner_radius[corner_index];
					color = *outer_color;
					corner_point = outer_points[corner_index];
				}
				float x = radius * (float)cos((double)corner_index * Math_PI / 2.0 + (double)detail / (double)adapted_corner_detail * Math_PI / 2.0 + Math_PI) + corner_point.x;
				float y = radius * (float)sin((double)corner_index * Math_PI / 2.0 + (double)detail / (double)adapted_corner_detail * Math_PI / 2.0 + Math_PI) + corner_point.y;
				verts.push_back(Vector2(x, y));
				colors.push_back(color);
			}
		}
	}

	if (rings == 2) {
		int vert_count = (adapted_corner_detail + 1) * 4 * rings;
		//fill the indices and the colors for the border
		for (int i = 0; i < vert_count; i++) {
			//poly 1
			indices.push_back(vert_offset + ((i + 0) % vert_count));
			indices.push_back(vert_offset + ((i + 2) % vert_count));
			indices.push_back(vert_offset + ((i + 1) % vert_count));
			//poly 2
			indices.push_back(vert_offset + ((i + 1) % vert_count));
			indices.push_back(vert_offset + ((i + 2) % vert_count));
			indices.push_back(vert_offset + ((i + 3) % vert_count));
		}
	}
}
