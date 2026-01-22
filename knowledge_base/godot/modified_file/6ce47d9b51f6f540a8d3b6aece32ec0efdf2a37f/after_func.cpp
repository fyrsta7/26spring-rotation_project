void ParallaxLayer::set_base_offset_and_scale(const Point2& p_offset,float p_scale) {

	if (!is_inside_tree())
		return;
	if (get_tree()->is_editor_hint())
		return;
	Point2 new_ofs = ((orig_offset+p_offset)*motion_scale)*p_scale+motion_offset;

	if (mirroring.x) {
		double den = mirroring.x*p_scale;
		new_ofs.x -= den*ceil(new_ofs.x/den);
	}

	if (mirroring.y) {
		double den = mirroring.y*p_scale;
		new_ofs.y -= den*ceil(new_ofs.y/den);
	}

	set_pos(new_ofs);
	set_scale(Vector2(1,1)*p_scale);


}
