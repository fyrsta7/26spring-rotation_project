void ParallaxLayer::set_base_offset_and_scale(const Point2& p_offset,float p_scale) {

	if (!is_inside_tree())
		return;
	if (get_tree()->is_editor_hint())
		return;
	Point2 new_ofs = ((orig_offset+p_offset)*motion_scale)*p_scale+motion_offset;

	if (mirroring.x) {

		while( new_ofs.x>=0) {
			new_ofs.x -= mirroring.x*p_scale;
		}
		while(new_ofs.x < -mirroring.x*p_scale) {
			new_ofs.x += mirroring.x*p_scale;
		}
	}

	if (mirroring.y) {

		while( new_ofs.y>=0) {
			new_ofs.y -= mirroring.y*p_scale;
		}
		while(new_ofs.y < -mirroring.y*p_scale) {
			new_ofs.y += mirroring.y*p_scale;
		}
	}


	set_pos(new_ofs);
	set_scale(Vector2(1,1)*p_scale);


}
