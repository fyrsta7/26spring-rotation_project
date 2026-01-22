void update_subtitles(sh_video_t *sh_video, double refpts, demux_stream_t *d_dvdsub, int reset)
{
    unsigned char *packet=NULL;
    int len;
    char type = d_dvdsub->sh ? ((sh_sub_t *)d_dvdsub->sh)->type : 'v';
    static subtitle subs;
    if (reset) {
        sub_clear_text(&subs, MP_NOPTS_VALUE);
        if (vo_sub) {
            set_osd_subtitle(NULL);
        }
        if (vo_spudec) {
            spudec_reset(vo_spudec);
            vo_osd_changed(OSDTYPE_SPU);
        }
    }
    // find sub
    if (subdata) {
        if (sub_fps==0) sub_fps = sh_video ? sh_video->fps : 25;
        current_module = "find_sub";
        if (refpts > sub_last_pts || refpts < sub_last_pts-1.0) {
            find_sub(subdata, (refpts+sub_delay) *
                     (subdata->sub_uses_time ? 100. : sub_fps));
            if (vo_sub) vo_sub_last = vo_sub;
            // FIXME! frame counter...
            sub_last_pts = refpts;
        }
    }

    // DVD sub:
    if (vo_config_count && vo_spudec &&
        (vobsub_id >= 0 || (dvdsub_id >= 0 && type == 'v'))) {
        int timestamp;
        current_module = "spudec";
        spudec_heartbeat(vo_spudec, 90000*sh_video->timer);
        /* Get a sub packet from the DVD or a vobsub and make a timestamp
         * relative to sh_video->timer */
        while(1) {
            // Vobsub
            len = 0;
            if (vo_vobsub) {
                if (refpts+sub_delay >= 0) {
                    len = vobsub_get_packet(vo_vobsub, refpts+sub_delay,
                                            (void**)&packet, &timestamp);
                    if (len > 0) {
                        timestamp -= (refpts + sub_delay - sh_video->timer)*90000;
                        mp_dbg(MSGT_CPLAYER,MSGL_V,"\rVOB sub: len=%d v_pts=%5.3f v_timer=%5.3f sub=%5.3f ts=%d \n",len,refpts,sh_video->timer,timestamp / 90000.0,timestamp);
                    }
                }
            } else {
                // DVD sub
                len = ds_get_packet_sub(d_dvdsub, (unsigned char**)&packet);
                if (len > 0) {
                    // XXX This is wrong, sh_video->pts can be arbitrarily
                    // much behind demuxing position. Unfortunately using
                    // d_video->pts which would have been the simplest
                    // improvement doesn't work because mpeg specific hacks
                    // in video.c set d_video->pts to 0.
                    float x = d_dvdsub->pts - refpts;
                    if (x > -20 && x < 20) // prevent missing subs on pts reset
                        timestamp = 90000*(sh_video->timer + d_dvdsub->pts
                                           + sub_delay - refpts);
                    else timestamp = 90000*(sh_video->timer + sub_delay);
                    mp_dbg(MSGT_CPLAYER, MSGL_V, "\rDVD sub: len=%d  "
                           "v_pts=%5.3f  s_pts=%5.3f  ts=%d \n", len,
                           refpts, d_dvdsub->pts, timestamp);
                }
            }
            if (len<=0 || !packet) break;
            if (vo_vobsub || timestamp >= 0)
                spudec_assemble(vo_spudec, packet, len, timestamp);
        }

        if (spudec_changed(vo_spudec))
            vo_osd_changed(OSDTYPE_SPU);
    } else if (dvdsub_id >= 0 && (type == 't' || type == 'm' || type == 'a' || type == 'd')) {
        double curpts = refpts + sub_delay;
        double endpts;
        if (type == 'd' && !d_dvdsub->demuxer->teletext) {
            tt_stream_props tsp = {0};
            void *ptr = &tsp;
            if (teletext_control(NULL, TV_VBI_CONTROL_START, &ptr) == VBI_CONTROL_TRUE)
                d_dvdsub->demuxer->teletext = ptr;
        }
        if (d_dvdsub->non_interleaved)
            ds_get_next_pts(d_dvdsub);
        while (d_dvdsub->first) {
            double subpts = ds_get_next_pts(d_dvdsub);
            if (subpts > curpts)
                break;
            endpts = d_dvdsub->first->endpts;
            len = ds_get_packet_sub(d_dvdsub, &packet);
            if (type == 'm') {
                if (len < 2) continue;
                len = FFMIN(len - 2, AV_RB16(packet));
                packet += 2;
            }
            if (type == 'd') {
                if (d_dvdsub->demuxer->teletext) {
                    uint8_t *p = packet;
                    p++;
                    len--;
                    while (len >= 46) {
                        int sublen = p[1];
                        if (p[0] == 2 || p[0] == 3)
                            teletext_control(d_dvdsub->demuxer->teletext,
                                TV_VBI_CONTROL_DECODE_DVB, p + 2);
                        p   += sublen + 2;
                        len -= sublen + 2;
                    }
                }
                continue;
            }
#ifdef CONFIG_ASS
            if (ass_enabled) {
                sh_sub_t* sh = d_dvdsub->sh;
                ass_track = sh ? sh->ass_track : NULL;
                if (!ass_track) continue;
                if (type == 'a') { // ssa/ass subs with libass
                    ass_process_chunk(ass_track, packet, len,
                                      (long long)(subpts*1000 + 0.5),
                                      (long long)((endpts-subpts)*1000 + 0.5));
                } else { // plaintext subs with libass
                    if (subpts != MP_NOPTS_VALUE) {
                        if (endpts == MP_NOPTS_VALUE) endpts = subpts + 3;
                        sub_clear_text(&subs, MP_NOPTS_VALUE);
                        sub_add_text(&subs, packet, len, endpts);
                        subs.start = subpts * 100;
                        subs.end = endpts * 100;
                        ass_process_subtitle(ass_track, &subs);
                    }
                }
                continue;
            }
#endif
            if (subpts != MP_NOPTS_VALUE) {
                if (endpts == MP_NOPTS_VALUE)
                    sub_clear_text(&subs, MP_NOPTS_VALUE);
                if (type == 'a') { // ssa/ass subs without libass => convert to plaintext
                    int i;
                    unsigned char* p = packet;
                    for (i=0; i < 8 && *p != '\0'; p++)
                        if (*p == ',')
                            i++;
                    if (*p == '\0')  /* Broken line? */
                        continue;
                    len -= p - packet;
                    packet = p;
                }
                sub_add_text(&subs, packet, len, endpts);
                set_osd_subtitle(&subs);
            }
            if (d_dvdsub->non_interleaved)
                ds_get_next_pts(d_dvdsub);
        }
        if (sub_clear_text(&subs, curpts))
            set_osd_subtitle(&subs);
    }
    current_module=NULL;
