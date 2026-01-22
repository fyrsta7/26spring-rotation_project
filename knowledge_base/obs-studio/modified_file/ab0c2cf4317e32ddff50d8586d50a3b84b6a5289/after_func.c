	                source->context.name, diff, expected, ts);

	/* if has video, ignore audio data until reset */
	if (!(source->info.output_flags & OBS_SOURCE_ASYNC))
		reset_audio_timing(source, ts, os_time);
}

#define VOL_MIN -96.0f
#define VOL_MAX  0.0f

static inline float to_db(float val)
{
	float db = 20.0f * log10f(val);
	return isfinite(db) ? db : VOL_MIN;
}

static void calc_volume_levels(struct obs_source *source, float *array,
		size_t frames, float volume)
{
	float sum_val = 0.0f;
	float max_val = 0.0f;
	float rms_val = 0.0f;

	audio_t        *audio          = obs_get_audio();
	const uint32_t sample_rate    = audio_output_get_sample_rate(audio);
	const size_t   channels       = audio_output_get_channels(audio);
	const size_t   count          = frames * channels;
	const size_t   vol_peak_delay = sample_rate * 3;
	const float    alpha          = 0.15f;

	for (size_t i = 0; i < count; i++) {
		float val      = array[i];
		float val_pow2 = val * val;

		sum_val += val_pow2;
		max_val  = (max_val > val_pow2) ? max_val : val_pow2;
	}

	/*
	  We want the volume meters scale linearly in respect to current
	  volume, so, no need to apply volume here.
	*/

	UNUSED_PARAMETER(volume);

