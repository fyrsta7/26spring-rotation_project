
	} else if (info->format == VIDEO_FORMAT_NV12) {
		compress_uyvx_to_nv12(
				input->data[0], input->linesize[0],
				0, info->height,
				output->data, output->linesize);

	} else if (info->format == VIDEO_FORMAT_I444) {
		convert_uyvx_to_i444(
				input->data[0], input->linesize[0],
				0, info->height,
				output->data, output->linesize);

