static inline void reset_frame_interval(struct game_capture *gc)
{
	struct obs_video_info ovi;
	uint64_t interval = 0;

	if (gc->config.limit_framerate && obs_get_video_info(&ovi))
		interval = ovi.fps_den * 1000000000ULL / ovi.fps_num;

	gc->global_hook_info->frame_interval = interval;
}
