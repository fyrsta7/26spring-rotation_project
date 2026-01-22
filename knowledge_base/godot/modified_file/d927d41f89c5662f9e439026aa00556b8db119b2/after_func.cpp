void CanvasItemEditor::_draw_ruler_tool() {
	if (tool != TOOL_RULER) {
		return;
	}

	const Ref<Texture2D> position_icon = get_editor_theme_icon(SNAME("EditorPosition"));
	if (ruler_tool_active) {
		Color ruler_primary_color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
		Color ruler_secondary_color = ruler_primary_color;
		ruler_secondary_color.a = 0.5;

		Point2 begin = (ruler_tool_origin - view_offset) * zoom;
		Point2 end = snap_point(viewport->get_local_mouse_position() / zoom + view_offset) * zoom - view_offset * zoom;
		Point2 corner = Point2(begin.x, end.y);
		Vector2 length_vector = (begin - end).abs() / zoom;

		const real_t horizontal_angle_rad = length_vector.angle();
		const real_t vertical_angle_rad = Math_PI / 2.0 - horizontal_angle_rad;

		Ref<Font> font = get_theme_font(SNAME("bold"), EditorStringName(EditorFonts));
		int font_size = 1.3 * get_theme_font_size(SNAME("bold_size"), EditorStringName(EditorFonts));
		Color font_color = get_theme_color(SNAME("font_color"), EditorStringName(Editor));
		Color font_secondary_color = font_color;
		font_secondary_color.set_v(font_secondary_color.get_v() > 0.5 ? 0.7 : 0.3);
		Color outline_color = font_color.inverted();
		float text_height = font->get_height(font_size);

		const float outline_size = 4;
		const float text_width = 76;
		const float angle_text_width = 54;

		Point2 text_pos = (begin + end) / 2 - Vector2(text_width / 2, text_height / 2);
		text_pos.x = CLAMP(text_pos.x, text_width / 2, viewport->get_rect().size.x - text_width * 1.5);
		text_pos.y = CLAMP(text_pos.y, text_height * 1.5, viewport->get_rect().size.y - text_height * 1.5);

		// Draw lines.
		viewport->draw_line(begin, end, ruler_primary_color, Math::round(EDSCALE * 3));

		bool draw_secondary_lines = !(Math::is_equal_approx(begin.y, corner.y) || Math::is_equal_approx(end.x, corner.x));
		if (draw_secondary_lines) {
			viewport->draw_line(begin, corner, ruler_secondary_color, Math::round(EDSCALE));
			viewport->draw_line(corner, end, ruler_secondary_color, Math::round(EDSCALE));

			// Angle arcs.
			int arc_point_count = 8;
			real_t arc_radius_max_length_percent = 0.1;
			real_t ruler_length = length_vector.length() * zoom;
			real_t arc_max_radius = 50.0;
			real_t arc_line_width = 2.0;

			const Vector2 end_to_begin = (end - begin);

			real_t arc_1_start_angle = end_to_begin.x < 0
					? (end_to_begin.y < 0 ? 3.0 * Math_PI / 2.0 - vertical_angle_rad : Math_PI / 2.0)
					: (end_to_begin.y < 0 ? 3.0 * Math_PI / 2.0 : Math_PI / 2.0 - vertical_angle_rad);
			real_t arc_1_end_angle = arc_1_start_angle + vertical_angle_rad;
			// Constrain arc to triangle height & max size.
			real_t arc_1_radius = MIN(MIN(arc_radius_max_length_percent * ruler_length, ABS(end_to_begin.y)), arc_max_radius);

			real_t arc_2_start_angle = end_to_begin.x < 0
					? (end_to_begin.y < 0 ? 0.0 : -horizontal_angle_rad)
					: (end_to_begin.y < 0 ? Math_PI - horizontal_angle_rad : Math_PI);
			real_t arc_2_end_angle = arc_2_start_angle + horizontal_angle_rad;
			// Constrain arc to triangle width & max size.
			real_t arc_2_radius = MIN(MIN(arc_radius_max_length_percent * ruler_length, ABS(end_to_begin.x)), arc_max_radius);

			viewport->draw_arc(begin, arc_1_radius, arc_1_start_angle, arc_1_end_angle, arc_point_count, ruler_primary_color, Math::round(EDSCALE * arc_line_width));
			viewport->draw_arc(end, arc_2_radius, arc_2_start_angle, arc_2_end_angle, arc_point_count, ruler_primary_color, Math::round(EDSCALE * arc_line_width));
		}

		// Draw text.
		if (begin.is_equal_approx(end)) {
			viewport->draw_string_outline(font, text_pos, (String)ruler_tool_origin, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
			viewport->draw_string(font, text_pos, (String)ruler_tool_origin, HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color);
			viewport->draw_texture(position_icon, (ruler_tool_origin - view_offset) * zoom - position_icon->get_size() / 2);
			return;
		}

		viewport->draw_string_outline(font, text_pos, TS->format_number(vformat("%.1f px", length_vector.length())), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
		viewport->draw_string(font, text_pos, TS->format_number(vformat("%.1f px", length_vector.length())), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color);

		if (draw_secondary_lines) {
			const int horizontal_angle = round(180 * horizontal_angle_rad / Math_PI);
			const int vertical_angle = round(180 * vertical_angle_rad / Math_PI);

			Point2 text_pos2 = text_pos;
			text_pos2.x = begin.x < text_pos.x ? MIN(text_pos.x - text_width, begin.x - text_width / 2) : MAX(text_pos.x + text_width, begin.x - text_width / 2);
			viewport->draw_string_outline(font, text_pos2, TS->format_number(vformat("%.1f px", length_vector.y)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
			viewport->draw_string(font, text_pos2, TS->format_number(vformat("%.1f px", length_vector.y)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color);

			Point2 v_angle_text_pos;
			v_angle_text_pos.x = CLAMP(begin.x - angle_text_width / 2, angle_text_width / 2, viewport->get_rect().size.x - angle_text_width);
			v_angle_text_pos.y = begin.y < end.y ? MIN(text_pos2.y - 2 * text_height, begin.y - text_height * 0.5) : MAX(text_pos2.y + text_height * 3, begin.y + text_height * 1.5);
			viewport->draw_string_outline(font, v_angle_text_pos, TS->format_number(vformat(U"%d°", vertical_angle)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
			viewport->draw_string(font, v_angle_text_pos, TS->format_number(vformat(U"%d°", vertical_angle)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color);

			text_pos2 = text_pos;
			text_pos2.y = end.y < text_pos.y ? MIN(text_pos.y - text_height * 2, end.y - text_height / 2) : MAX(text_pos.y + text_height * 2, end.y - text_height / 2);
			viewport->draw_string_outline(font, text_pos2, TS->format_number(vformat("%.1f px", length_vector.x)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
			viewport->draw_string(font, text_pos2, TS->format_number(vformat("%.1f px", length_vector.x)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color);

			Point2 h_angle_text_pos;
			h_angle_text_pos.x = CLAMP(end.x - angle_text_width / 2, angle_text_width / 2, viewport->get_rect().size.x - angle_text_width);
			if (begin.y < end.y) {
				h_angle_text_pos.y = end.y + text_height * 1.5;
				if (ABS(text_pos2.x - h_angle_text_pos.x) < text_width) {
					int height_multiplier = 1.5 + (int)grid_snap_active;
					h_angle_text_pos.y = MAX(text_pos.y + height_multiplier * text_height, MAX(end.y + text_height * 1.5, text_pos2.y + height_multiplier * text_height));
				}
			} else {
				h_angle_text_pos.y = end.y - text_height * 0.5;
				if (ABS(text_pos2.x - h_angle_text_pos.x) < text_width) {
					int height_multiplier = 1 + (int)grid_snap_active;
					h_angle_text_pos.y = MIN(text_pos.y - height_multiplier * text_height, MIN(end.y - text_height * 0.5, text_pos2.y - height_multiplier * text_height));
				}
			}
			viewport->draw_string_outline(font, h_angle_text_pos, TS->format_number(vformat(U"%d°", horizontal_angle)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
			viewport->draw_string(font, h_angle_text_pos, TS->format_number(vformat(U"%d°", horizontal_angle)), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color);
		}

		if (grid_snap_active) {
			text_pos = (begin + end) / 2 + Vector2(-text_width / 2, text_height / 2);
			text_pos.x = CLAMP(text_pos.x, text_width / 2, viewport->get_rect().size.x - text_width * 1.5);
			text_pos.y = CLAMP(text_pos.y, text_height * 2.5, viewport->get_rect().size.y - text_height / 2);

			if (draw_secondary_lines) {
				viewport->draw_string_outline(font, text_pos, TS->format_number(vformat("%.2f " + TTR("units"), (length_vector / grid_step).length())), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
				viewport->draw_string(font, text_pos, TS->format_number(vformat("%.2f " + TTR("units"), (length_vector / grid_step).length())), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color);

				Point2 text_pos2 = text_pos;
				text_pos2.x = begin.x < text_pos.x ? MIN(text_pos.x - text_width, begin.x - text_width / 2) : MAX(text_pos.x + text_width, begin.x - text_width / 2);
				viewport->draw_string_outline(font, text_pos2, TS->format_number(vformat("%d " + TTR("units"), roundf(length_vector.y / grid_step.y))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
				viewport->draw_string(font, text_pos2, TS->format_number(vformat("%d " + TTR("units"), roundf(length_vector.y / grid_step.y))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color);

				text_pos2 = text_pos;
				text_pos2.y = end.y < text_pos.y ? MIN(text_pos.y - text_height * 2, end.y + text_height / 2) : MAX(text_pos.y + text_height * 2, end.y + text_height / 2);
				viewport->draw_string_outline(font, text_pos2, TS->format_number(vformat("%d " + TTR("units"), roundf(length_vector.x / grid_step.x))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
				viewport->draw_string(font, text_pos2, TS->format_number(vformat("%d " + TTR("units"), roundf(length_vector.x / grid_step.x))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_secondary_color);
			} else {
				viewport->draw_string_outline(font, text_pos, TS->format_number(vformat("%d " + TTR("units"), roundf((length_vector / grid_step).length()))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, outline_size, outline_color);
				viewport->draw_string(font, text_pos, TS->format_number(vformat("%d " + TTR("units"), roundf((length_vector / grid_step).length()))), HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, font_color);
			}
		}
	} else {
		if (grid_snap_active) {
			viewport->draw_texture(position_icon, (ruler_tool_origin - view_offset) * zoom - position_icon->get_size() / 2);
		}
	}
}
