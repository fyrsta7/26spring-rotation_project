
static int video_feed_async_filter(struct MPContext *mpctx)
{
    struct vf_chain *vf = mpctx->vo_chain->vf;

    if (vf->initialized < 0)
        return VD_ERROR;

    if (vf_needs_input(vf) < 1)
        return 0;
    mp_wakeup_core(mpctx); // retry until done
    return video_decode_and_filter(mpctx);
}

/* Modify video timing to match the audio timeline. There are two main
 * reasons this is needed. First, video and audio can start from different
 * positions at beginning of file or after a seek (MPlayer starts both
 * immediately even if they have different pts). Second, the file can have
 * audio timestamps that are inconsistent with the duration of the audio
 * packets, for example two consecutive timestamp values differing by
 * one second but only a packet with enough samples for half a second
 * of playback between them.
 */
