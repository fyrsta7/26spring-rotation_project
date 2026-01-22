double AnimationNodeBlendSpace1D::process(double p_time, bool p_seek, bool p_is_external_seeking) {
	if (blend_points_used == 0) {
		return 0.0;
	}

	if (blend_points_used == 1) {
		// only one point available, just play that animation
		return blend_node(blend_points[0].name, blend_points[0].node, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
	}

	double blend_pos = get_parameter(blend_position);
	int cur_closest = get_parameter(closest);
	double cur_length_internal = get_parameter(length_internal);
	double max_time_remaining = 0.0;

	if (blend_mode == BLEND_MODE_INTERPOLATED) {
		int point_lower = -1;
		float pos_lower = 0.0;
		int point_higher = -1;
		float pos_higher = 0.0;

		// find the closest two points to blend between
		for (int i = 0; i < blend_points_used; i++) {
			float pos = blend_points[i].position;

			if (pos <= blend_pos) {
				if (point_lower == -1 || pos > pos_lower) {
					point_lower = i;
					pos_lower = pos;
				}
			} else if (point_higher == -1 || pos < pos_higher) {
				point_higher = i;
				pos_higher = pos;
			}
		}

		// fill in weights
		float weights[MAX_BLEND_POINTS] = {};
		if (point_lower == -1 && point_higher != -1) {
			// we are on the left side, no other point to the left
			// we just play the next point.

			weights[point_higher] = 1.0;
		} else if (point_higher == -1) {
			// we are on the right side, no other point to the right
			// we just play the previous point

			weights[point_lower] = 1.0;
		} else {
			// we are between two points.
			// figure out weights, then blend the animations

			float distance_between_points = pos_higher - pos_lower;

			float current_pos_inbetween = blend_pos - pos_lower;

			float blend_percentage = current_pos_inbetween / distance_between_points;

			float blend_lower = 1.0 - blend_percentage;
			float blend_higher = blend_percentage;

			weights[point_lower] = blend_lower;
			weights[point_higher] = blend_higher;
		}

		// actually blend the animations now

		for (int i = 0; i < blend_points_used; i++) {
			if (i == point_lower || i == point_higher) {
				double remaining = blend_node(blend_points[i].name, blend_points[i].node, p_time, p_seek, p_is_external_seeking, weights[i], FILTER_IGNORE, true);
				max_time_remaining = MAX(max_time_remaining, remaining);
			} else if (sync) {
				blend_node(blend_points[i].name, blend_points[i].node, p_time, p_seek, p_is_external_seeking, 0, FILTER_IGNORE, true);
			}
		}
	} else {
		int new_closest = -1;
		double new_closest_dist = 1e20;

		for (int i = 0; i < blend_points_used; i++) {
			double d = abs(blend_points[i].position - blend_pos);
			if (d < new_closest_dist) {
				new_closest = i;
				new_closest_dist = d;
			}
		}

		if (new_closest != cur_closest && new_closest != -1) {
			double from = 0.0;
			if (blend_mode == BLEND_MODE_DISCRETE_CARRY && cur_closest != -1) {
				//for ping-pong loop
				Ref<AnimationNodeAnimation> na_c = static_cast<Ref<AnimationNodeAnimation>>(blend_points[cur_closest].node);
				Ref<AnimationNodeAnimation> na_n = static_cast<Ref<AnimationNodeAnimation>>(blend_points[new_closest].node);
				if (!na_c.is_null() && !na_n.is_null()) {
					na_n->set_backward(na_c->is_backward());
				}
				//see how much animation remains
				from = cur_length_internal - blend_node(blend_points[cur_closest].name, blend_points[cur_closest].node, p_time, false, p_is_external_seeking, 0.0, FILTER_IGNORE, true);
			}

			max_time_remaining = blend_node(blend_points[new_closest].name, blend_points[new_closest].node, from, true, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
			cur_length_internal = from + max_time_remaining;

			cur_closest = new_closest;

		} else {
			max_time_remaining = blend_node(blend_points[cur_closest].name, blend_points[cur_closest].node, p_time, p_seek, p_is_external_seeking, 1.0, FILTER_IGNORE, true);
		}

		if (sync) {
			for (int i = 0; i < blend_points_used; i++) {
				if (i != cur_closest) {
					blend_node(blend_points[i].name, blend_points[i].node, p_time, p_seek, p_is_external_seeking, 0, FILTER_IGNORE, true);
				}
			}
		}
	}

	set_parameter(this->closest, cur_closest);
	set_parameter(this->length_internal, cur_length_internal);
	return max_time_remaining;
}
