	float vol;

	pthread_mutex_lock(&source->audio_actions_mutex);

	actions_pending = source->audio_actions.num > 0;
	if (actions_pending)
		action = source->audio_actions.array[0];

	pthread_mutex_unlock(&source->audio_actions_mutex);

	if (actions_pending) {
		uint64_t duration = conv_frames_to_time(sample_rate,
				AUDIO_OUTPUT_FRAMES);

		if (action.timestamp < (source->audio_ts + duration)) {
			apply_audio_actions(source, channels, sample_rate);
			return;
		}
	}

	vol = get_source_volume(source, source->audio_ts);
	if (vol == 1.0f)
		return;

	if (vol == 0.0f || mixers == 0) {
		memset(source->audio_output_buf[0][0], 0,
				AUDIO_OUTPUT_FRAMES * sizeof(float) *
				MAX_AUDIO_CHANNELS * MAX_AUDIO_MIXES);
		return;
	}

	for (size_t mix = 0; mix < MAX_AUDIO_MIXES; mix++) {
		uint32_t mix_and_val = (1 << mix);
		if ((source->audio_mixers & mix_and_val) != 0 &&
		    (mixers & mix_and_val) != 0)
			multiply_output_audio(source, mix, channels, vol);
	}
}

static void custom_audio_render(obs_source_t *source, uint32_t mixers,
		size_t channels, size_t sample_rate)
{
	struct obs_source_audio_mix audio_data;
