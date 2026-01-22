void ParallaxLayer::set_base_offset_and_scale(const Point2& p_offset,float p_scale) {

	if (!is_inside_tree())
		return;
	if (get_tree()->is_editor_hint())
		return;
	Point2 new_ofs = ((orig_offset+p_offset)*motion_scale)*p_scale;

	if (mirroring.x) {
		double den = mirroring.x*p_scale;
		new_ofs.x = fmod(new_ofs.x,den) - (mirroring.x > 0 ? den : 0);
	}

	if (mirroring.y) {
		double den = mirroring.y*p_scale;
		new_ofs.y = fmod(new_ofs.y,den) - (mirroring.y > 0 ? den : 0);
	}


	set_pos(new_ofs);
	set_scale(Vector2(1,1)*p_scale);


}
