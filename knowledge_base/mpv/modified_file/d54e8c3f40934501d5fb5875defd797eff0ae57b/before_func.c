            if (component_tag >= 0x30 && component_tag <= 0x37)
                lav->profile = FF_PROFILE_ARIB_PROFILE_A;
            break;
        case 0x0012:
            // component tag 0x87 signifies a mobile/partial reception
            // (1seg) captioning service ("profile C").
            if (component_tag == 0x87)
                lav->profile = FF_PROFILE_ARIB_PROFILE_C;
            break;
        }
        if (lav->profile == FF_PROFILE_UNKNOWN)
            MP_WARN(demuxer, "ARIB caption profile %02x / %04x not supported.\n",
                    component_tag, data_component_id);
    }

    demux_add_sh_stream(demuxer, sh);

    if (!subtitle_type)
        MP_ERR(demuxer, "Subtitle type '%s' is not supported.\n", track->codec_id);

    return 0;
}

// Workaround for broken files that don't set attached_picture
static void probe_if_image(demuxer_t *demuxer)
{
    mkv_demuxer_t *mkv_d = demuxer->priv;

    for (int n = 0; n < mkv_d->num_tracks; n++) {
        int video_blocks = 0;
        mkv_track_t *track = mkv_d->tracks[n];
