void RendererCanvasCull::_cull_canvas_item(Item *p_canvas_item, const Transform2D &p_parent_xform, const Rect2 &p_clip_rect, const Color &p_modulate, int p_z, RendererCanvasRender::Item **r_z_list, RendererCanvasRender::Item **r_z_last_list, Item *p_canvas_clip, Item *p_material_owner, bool p_is_already_y_sorted, uint32_t p_canvas_cull_mask, const Point2 &p_repeat_size, int p_repeat_times, RendererCanvasRender::Item *p_repeat_source_item) {
	Item *ci = p_canvas_item;

	if (!ci->visible) {
		return;
	}

	if (!(ci->visibility_layer & p_canvas_cull_mask)) {
		return;
	}

	if (ci->children_order_dirty) {
		ci->child_items.sort_custom<ItemIndexSort>();
		ci->children_order_dirty = false;
	}

	Rect2 rect = ci->get_rect();

	if (ci->visibility_notifier) {
		if (ci->visibility_notifier->area.size != Vector2()) {
			rect = rect.merge(ci->visibility_notifier->area);
		}
	}

	Transform2D self_xform;
	Transform2D final_xform;
	if (p_is_already_y_sorted) {
		// Y-sorted item's final transform is calculated before y-sorting,
		// and is passed as `p_parent_xform` afterwards. No need to recalculate.
		final_xform = p_parent_xform;
	} else {
		if (!_interpolation_data.interpolation_enabled || !ci->interpolated) {
			self_xform = ci->xform_curr;
		} else {
			real_t f = Engine::get_singleton()->get_physics_interpolation_fraction();
			TransformInterpolator::interpolate_transform_2d(ci->xform_prev, ci->xform_curr, self_xform, f);
		}

		Transform2D parent_xform = p_parent_xform;

		if (snapping_2d_transforms_to_pixel) {
			self_xform.columns[2] = (self_xform.columns[2] + Point2(0.5, 0.5)).floor();
			parent_xform.columns[2] = (parent_xform.columns[2] + Point2(0.5, 0.5)).floor();
		}

		final_xform = parent_xform * self_xform;
	}

	Point2 repeat_size = p_repeat_size;
	int repeat_times = p_repeat_times;
	RendererCanvasRender::Item *repeat_source_item = p_repeat_source_item;

	if (ci->repeat_source) {
		repeat_size = ci->repeat_size;
		repeat_times = ci->repeat_times;
		repeat_source_item = ci;
	} else {
		ci->repeat_size = repeat_size;
		ci->repeat_times = repeat_times;
		ci->repeat_source_item = repeat_source_item;
	}

	Rect2 global_rect = final_xform.xform(rect);
	if (repeat_source_item && (repeat_size.x || repeat_size.y)) {
		// Top-left repeated rect.
		Rect2 corner_rect = global_rect;
		corner_rect.position -= repeat_source_item->final_transform.basis_xform((repeat_times / 2) * repeat_size);
		global_rect = corner_rect;

		// Plus top-right repeated rect.
		Size2 size_x_offset = repeat_source_item->final_transform.basis_xform(repeat_times * Size2(repeat_size.x, 0));
		corner_rect.position += size_x_offset;
		global_rect = global_rect.merge(corner_rect);

		// Plus bottom-right repeated rect.
		corner_rect.position += repeat_source_item->final_transform.basis_xform(repeat_times * Size2(0, repeat_size.y));
		global_rect = global_rect.merge(corner_rect);

		// Plus bottom-left repeated rect.
		corner_rect.position -= size_x_offset;
		global_rect = global_rect.merge(corner_rect);
	}
	global_rect.position += p_clip_rect.position;

	if (ci->use_parent_material && p_material_owner) {
		ci->material_owner = p_material_owner;
	} else {
		p_material_owner = ci;
		ci->material_owner = nullptr;
	}

	Color modulate(ci->modulate.r * p_modulate.r, ci->modulate.g * p_modulate.g, ci->modulate.b * p_modulate.b, ci->modulate.a * p_modulate.a);

	if (modulate.a < 0.007) {
		return;
	}

	int child_item_count = ci->child_items.size();
	Item **child_items = ci->child_items.ptrw();

	if (ci->clip) {
		if (p_canvas_clip != nullptr) {
			ci->final_clip_rect = p_canvas_clip->final_clip_rect.intersection(global_rect);
		} else {
			ci->final_clip_rect = p_clip_rect.intersection(global_rect);
		}
		if (ci->final_clip_rect.size.width < 0.5 || ci->final_clip_rect.size.height < 0.5) {
			// The clip rect area is 0, so don't draw the item.
			return;
		}
		ci->final_clip_rect.position = ci->final_clip_rect.position.round();
		ci->final_clip_rect.size = ci->final_clip_rect.size.round();
		ci->final_clip_owner = ci;

	} else {
		ci->final_clip_owner = p_canvas_clip;
	}

	int parent_z = p_z;
	if (ci->z_relative) {
		p_z = CLAMP(p_z + ci->z_index, RS::CANVAS_ITEM_Z_MIN, RS::CANVAS_ITEM_Z_MAX);
	} else {
		p_z = ci->z_index;
	}

	if (ci->sort_y) {
		if (!p_is_already_y_sorted) {
			if (ci->ysort_children_count == -1) {
				ci->ysort_children_count = _count_ysort_children(ci);
			}

			child_item_count = ci->ysort_children_count + 1;
			child_items = (Item **)alloca(child_item_count * sizeof(Item *));

			ci->ysort_xform = Transform2D();
			ci->ysort_modulate = Color(1, 1, 1, 1);
			ci->ysort_index = 0;
			ci->ysort_parent_abs_z_index = parent_z;
			child_items[0] = ci;
			int i = 1;
			_collect_ysort_children(ci, p_material_owner, Color(1, 1, 1, 1), child_items, i, p_z);

			SortArray<Item *, ItemYSort> sorter;
			sorter.sort(child_items, child_item_count);

			for (i = 0; i < child_item_count; i++) {
				_cull_canvas_item(child_items[i], final_xform * child_items[i]->ysort_xform, p_clip_rect, modulate * child_items[i]->ysort_modulate, child_items[i]->ysort_parent_abs_z_index, r_z_list, r_z_last_list, (Item *)ci->final_clip_owner, (Item *)child_items[i]->material_owner, true, p_canvas_cull_mask, child_items[i]->repeat_size, child_items[i]->repeat_times, child_items[i]->repeat_source_item);
			}
		} else {
			RendererCanvasRender::Item *canvas_group_from = nullptr;
			bool use_canvas_group = ci->canvas_group != nullptr && (ci->canvas_group->fit_empty || ci->commands != nullptr);
			if (use_canvas_group) {
				int zidx = p_z - RS::CANVAS_ITEM_Z_MIN;
				canvas_group_from = r_z_last_list[zidx];
			}

			_attach_canvas_item_for_draw(ci, p_canvas_clip, r_z_list, r_z_last_list, final_xform, p_clip_rect, global_rect, modulate, p_z, p_material_owner, use_canvas_group, canvas_group_from);
		}
	} else {
		RendererCanvasRender::Item *canvas_group_from = nullptr;
		bool use_canvas_group = ci->canvas_group != nullptr && (ci->canvas_group->fit_empty || ci->commands != nullptr);
		if (use_canvas_group) {
			int zidx = p_z - RS::CANVAS_ITEM_Z_MIN;
			canvas_group_from = r_z_last_list[zidx];
		}

		for (int i = 0; i < child_item_count; i++) {
			if (!child_items[i]->behind && !use_canvas_group) {
				continue;
			}
			_cull_canvas_item(child_items[i], final_xform, p_clip_rect, modulate, p_z, r_z_list, r_z_last_list, (Item *)ci->final_clip_owner, p_material_owner, false, p_canvas_cull_mask, repeat_size, repeat_times, repeat_source_item);
		}
		_attach_canvas_item_for_draw(ci, p_canvas_clip, r_z_list, r_z_last_list, final_xform, p_clip_rect, global_rect, modulate, p_z, p_material_owner, use_canvas_group, canvas_group_from);
		for (int i = 0; i < child_item_count; i++) {
			if (child_items[i]->behind || use_canvas_group) {
				continue;
			}
			_cull_canvas_item(child_items[i], final_xform, p_clip_rect, modulate, p_z, r_z_list, r_z_last_list, (Item *)ci->final_clip_owner, p_material_owner, false, p_canvas_cull_mask, repeat_size, repeat_times, repeat_source_item);
		}
	}
}
