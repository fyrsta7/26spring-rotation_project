float DisplayServerX11::screen_get_refresh_rate(int p_screen) const {
	_THREAD_SAFE_METHOD_

	p_screen = _get_screen_index(p_screen);
	ERR_FAIL_INDEX_V(p_screen, get_screen_count(), SCREEN_REFRESH_RATE_FALLBACK);

	//Use xrandr to get screen refresh rate.
	if (xrandr_ext_ok) {
		XRRScreenResources *screen_info = XRRGetScreenResources(x11_display, windows[MAIN_WINDOW_ID].x11_window);
		if (screen_info) {
			RRMode current_mode = 0;
			xrr_monitor_info *monitors = nullptr;

			if (xrr_get_monitors) {
				int count = 0;
				monitors = xrr_get_monitors(x11_display, windows[MAIN_WINDOW_ID].x11_window, true, &count);
				ERR_FAIL_INDEX_V(p_screen, count, SCREEN_REFRESH_RATE_FALLBACK);
			} else {
				ERR_PRINT("An error occurred while trying to get the screen refresh rate.");
				return SCREEN_REFRESH_RATE_FALLBACK;
			}

			bool found_active_mode = false;
			for (int crtc = 0; crtc < screen_info->ncrtc; crtc++) { // Loop through outputs to find which one is currently outputting.
				XRRCrtcInfo *monitor_info = XRRGetCrtcInfo(x11_display, screen_info, screen_info->crtcs[crtc]);
				if (monitor_info->x != monitors[p_screen].x || monitor_info->y != monitors[p_screen].y) { // If X and Y aren't the same as the monitor we're looking for, this isn't the right monitor. Continue.
					continue;
				}

				if (monitor_info->mode != None) {
					current_mode = monitor_info->mode;
					found_active_mode = true;
					break;
				}
			}

			if (found_active_mode) {
				for (int mode = 0; mode < screen_info->nmode; mode++) {
					XRRModeInfo m_info = screen_info->modes[mode];
					if (m_info.id == current_mode) {
						// Snap to nearest 0.01 to stay consistent with other platforms.
						return Math::snapped((float)m_info.dotClock / ((float)m_info.hTotal * (float)m_info.vTotal), 0.01);
					}
				}
			}

			ERR_PRINT("An error occurred while trying to get the screen refresh rate."); // We should have returned the refresh rate by now. An error must have occurred.
			return SCREEN_REFRESH_RATE_FALLBACK;
		} else {
			ERR_PRINT("An error occurred while trying to get the screen refresh rate.");
			return SCREEN_REFRESH_RATE_FALLBACK;
		}
	}
	ERR_PRINT("An error occurred while trying to get the screen refresh rate.");
	return SCREEN_REFRESH_RATE_FALLBACK;
}
