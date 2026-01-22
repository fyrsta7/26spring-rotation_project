
        EsOutProgramEpgEvent( out, source, i_group, p_evt );
        return VLC_SUCCESS;
    }
    case ES_OUT_SET_EPG_TIME:
    {
        int64_t i64 = va_arg( args, int64_t );

        EsOutEpgTime( out, i64 );
        return VLC_SUCCESS;
    }

    case ES_OUT_DEL_GROUP:
    {
        int i_group = va_arg( args, int );

        return EsOutProgramDel( out, source, i_group );
    }

    case ES_OUT_SET_META:
    {
        const vlc_meta_t *p_meta = va_arg( args, const vlc_meta_t * );

        EsOutGlobalMeta( out, p_meta );
        return VLC_SUCCESS;
    }

    case ES_OUT_GET_EMPTY:
    {
        bool *pb = va_arg( args, bool* );
        *pb = EsOutDecodersIsEmpty( out );
        return VLC_SUCCESS;
    }

    case ES_OUT_GET_PCR_SYSTEM:
    {
        if( p_sys->b_buffering )
            return VLC_EGENERIC;

        es_out_pgrm_t *p_pgrm = p_sys->p_pgrm;
        if( !p_pgrm )
            return VLC_EGENERIC;

        vlc_tick_t *pi_system = va_arg( args, vlc_tick_t *);
        vlc_tick_t *pi_delay  = va_arg( args, vlc_tick_t *);
        input_clock_GetSystemOrigin( p_pgrm->p_input_clock, pi_system, pi_delay );
        return VLC_SUCCESS;
    }

    case ES_OUT_MODIFY_PCR_SYSTEM:
    {
        if( p_sys->b_buffering )
            return VLC_EGENERIC;

        es_out_pgrm_t *p_pgrm = p_sys->p_pgrm;
        if( !p_pgrm )
            return VLC_EGENERIC;

        const bool    b_absolute = va_arg( args, int );
        const vlc_tick_t i_system   = va_arg( args, vlc_tick_t );
        input_clock_ChangeSystemOrigin( p_pgrm->p_input_clock, b_absolute, i_system );
        return VLC_SUCCESS;
    }

    case ES_OUT_POST_SUBNODE:
    {
        input_thread_t *input = p_sys->p_input;
        input_item_node_t *node = va_arg(args, input_item_node_t *);
        input_SendEventParsing(input, node);
        input_item_node_Delete(node);

        return VLC_SUCCESS;
    }

    case ES_OUT_VOUT_SET_MOUSE_EVENT:
    {
        es_out_id_t *p_es = va_arg( args, es_out_id_t * );

        if( !p_es || p_es->fmt.i_cat != VIDEO_ES )
            return VLC_EGENERIC;

        p_es->mouse_event_cb = va_arg( args, vlc_mouse_event );
        p_es->mouse_event_userdata = va_arg( args, void * );

        if( p_es->p_dec )
            vlc_input_decoder_SetVoutMouseEvent( p_es->p_dec,
                p_es->mouse_event_cb, p_es->mouse_event_userdata );

        return VLC_SUCCESS;
    }
    case ES_OUT_VOUT_ADD_OVERLAY:
    {
        es_out_id_t *p_es = va_arg( args, es_out_id_t * );
        subpicture_t *sub = va_arg( args, subpicture_t * );
        size_t *channel = va_arg( args, size_t * );
        if( p_es && p_es->fmt.i_cat == VIDEO_ES && p_es->p_dec )
            return vlc_input_decoder_AddVoutOverlay( p_es->p_dec, sub, channel );
        return VLC_EGENERIC;
    }
    case ES_OUT_VOUT_DEL_OVERLAY:
    {
        es_out_id_t *p_es = va_arg( args, es_out_id_t * );
        size_t channel = va_arg( args, size_t );
        if( p_es && p_es->fmt.i_cat == VIDEO_ES && p_es->p_dec )
            return vlc_input_decoder_DelVoutOverlay( p_es->p_dec, channel );
        return VLC_EGENERIC;
    }
    case ES_OUT_SPU_SET_HIGHLIGHT:
    {
        es_out_id_t *p_es = va_arg( args, es_out_id_t * );
        const vlc_spu_highlight_t *spu_hl =
            va_arg( args, const vlc_spu_highlight_t * );
        if( p_es && p_es->fmt.i_cat == SPU_ES && p_es->p_dec )
            return vlc_input_decoder_SetSpuHighlight( p_es->p_dec, spu_hl );
        return VLC_EGENERIC;
    }
    default: vlc_assert_unreachable();
    }
}

static int EsOutVaPrivControlLocked( es_out_t *out, int query, va_list args )
{
    es_out_sys_t *p_sys = container_of(out, es_out_sys_t, out);

    switch (query)
    {
    case ES_OUT_PRIV_SET_MODE:
    {
        const int i_mode = va_arg( args, int );
        assert( i_mode == ES_OUT_MODE_NONE || i_mode == ES_OUT_MODE_ALL ||
                i_mode == ES_OUT_MODE_AUTO || i_mode == ES_OUT_MODE_PARTIAL ||
                i_mode == ES_OUT_MODE_END );

        if (i_mode != ES_OUT_MODE_NONE && !p_sys->b_active && !vlc_list_is_empty(&p_sys->es))
        {
            /* XXX Terminate vout if there are tracks but no video one.
             * This one is not mandatory but is he earliest place where it
             * can be done */
            es_out_id_t *p_es;
            bool found = false;

            foreach_es_then_es_slaves(p_es)
                if( p_es->fmt.i_cat == VIDEO_ES && !found /* nested loop */ )
                {
                    found = true;
                    break;
                }

            if (!found)
                EsOutStopFreeVout( out );
        }
        p_sys->b_active = i_mode != ES_OUT_MODE_NONE;
        p_sys->i_mode = i_mode;

        if( i_mode == ES_OUT_MODE_NONE )
        {
            /* Reset main clocks before unselecting every ESes. This will speed
             * up audio and video output termination. Indeed, they won't wait
             * for a specific PTS conversion. This may also unblock outputs in
             * case of a corrupted sample with a PTS very far in the future.
             * */
            es_out_pgrm_t *pgrm;
            vlc_list_foreach(pgrm, &p_sys->programs, node)
                vlc_clock_main_Reset(pgrm->p_main_clock);
        }

        /* Reapply policy mode */
        es_out_id_t *es;

        foreach_es_then_es_slaves(es)
        {
            if (EsIsSelected(es))
                EsOutUnselectEs(out, es, es->p_pgrm == p_sys->p_pgrm);
        }
        foreach_es_then_es_slaves(es)
        {
            EsOutSelect(out, es, false);
        }

        if( i_mode == ES_OUT_MODE_END )
            EsOutTerminate( out );
        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_SET_ES:
    case ES_OUT_PRIV_UNSET_ES:
    case ES_OUT_PRIV_RESTART_ES:
    {
        vlc_es_id_t *es_id = va_arg( args, vlc_es_id_t * );
        es_out_id_t *es = vlc_es_id_get_out( es_id );
        int new_query;
        switch( query )
        {
            case ES_OUT_PRIV_SET_ES: new_query = ES_OUT_SET_ES; break;
            case ES_OUT_PRIV_UNSET_ES: new_query = ES_OUT_UNSET_ES; break;
            case ES_OUT_PRIV_RESTART_ES: new_query = ES_OUT_RESTART_ES; break;
            default: vlc_assert_unreachable();
        }
        return EsOutControlLocked( out, p_sys->main_source, new_query, es );
    }
    case ES_OUT_PRIV_SET_ES_CAT_IDS:
    {
        enum es_format_category_e cat = va_arg( args, enum es_format_category_e );
        const char *str_ids = va_arg( args, const char * );
        es_out_es_props_t *p_esprops = GetPropsByCat( p_sys, cat );
        free( p_esprops->str_ids );
        p_esprops->str_ids = str_ids ? strdup( str_ids ) : NULL;

        if( p_esprops->str_ids )
        {
            /* Update new tracks selection using the new str_ids */
            EsOutSelectListFromProps( out, cat );
        }

        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_GET_WAKE_UP:
    {
        vlc_tick_t *pi_wakeup = va_arg( args, vlc_tick_t* );
        *pi_wakeup = EsOutGetWakeup( out );
        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_SET_ES_LIST:
    {
        enum es_format_category_e cat = va_arg( args, enum es_format_category_e );
        vlc_es_id_t *const*es_id_list = va_arg( args, vlc_es_id_t ** );
        EsOutSelectList( out, cat, es_id_list );
        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_STOP_ALL_ES:
    {
        es_out_id_t *es;
        int count = 0;

        foreach_es_then_es_slaves(es)
            count++;

        vlc_es_id_t **selected_es = vlc_alloc(count + 1, sizeof(vlc_es_id_t *));
        if (!selected_es)
            return VLC_ENOMEM;

        *va_arg(args, vlc_es_id_t ***) = selected_es;

        foreach_es_then_es_slaves(es)
        {
            if (EsIsSelected(es))
            {
                EsOutDestroyDecoder(out, es);
                *selected_es++ = vlc_es_id_Hold(&es->id);
            }
            *selected_es = NULL;
        }
        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_START_ALL_ES:
    {
        vlc_es_id_t **selected_es = va_arg( args, vlc_es_id_t ** );
        vlc_es_id_t **selected_es_it = selected_es;
        for( vlc_es_id_t *id = *selected_es_it; id != NULL;
             id = *++selected_es_it )
        {
            EsOutCreateDecoder( out, vlc_es_id_get_out( id ) );
            vlc_es_id_Release( id );
        }
        free(selected_es);
        EsOutStopFreeVout( out );
        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_GET_BUFFERING:
    {
        bool *pb = va_arg( args, bool* );
        *pb = p_sys->b_buffering;
        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_SET_ES_DELAY:
    {
        vlc_es_id_t *es_id = va_arg( args, vlc_es_id_t * );
        es_out_id_t *es = vlc_es_id_get_out( es_id );
        const vlc_tick_t delay = va_arg(args, vlc_tick_t);
        EsOutSetEsDelay(out, es, delay);
        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_SET_DELAY:
    {
        const int i_cat = va_arg( args, int );
        const vlc_tick_t i_delay = va_arg( args, vlc_tick_t );
        EsOutSetDelay( out, i_cat, i_delay );
        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_SET_RECORD_STATE:
    {
        bool b = va_arg( args, int );
        return EsOutSetRecord( out, b );
    }
    case ES_OUT_PRIV_SET_PAUSE_STATE:
    {
        const bool b_source_paused = (bool)va_arg( args, int );
        const bool b_paused = (bool)va_arg( args, int );
        const vlc_tick_t i_date = va_arg( args, vlc_tick_t );

        assert( !b_source_paused == !b_paused );
        EsOutChangePause( out, b_paused, i_date );

        return VLC_SUCCESS;
    }
    case ES_OUT_PRIV_SET_RATE:
    {
        const float src_rate = va_arg( args, double );
        const float rate = va_arg( args, double );
