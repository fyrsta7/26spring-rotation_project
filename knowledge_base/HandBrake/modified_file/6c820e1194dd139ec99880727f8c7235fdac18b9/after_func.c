    char             filename[1024];
};

/***********************************************************************
 * hb_work_encx264_init
 ***********************************************************************
 *
 **********************************************************************/
int encx264Init( hb_work_object_t * w, hb_job_t * job )
{
    x264_param_t       param;
    x264_nal_t       * nal;
    int                nal_count;

    hb_work_private_t * pv = calloc( 1, sizeof( hb_work_private_t ) );
    w->private_data = pv;

    pv->job = job;

    memset( pv->filename, 0, 1024 );
    hb_get_tempory_filename( job->h, pv->filename, "x264.log" );

    x264_param_default( &param );

    param.i_threads    = ( hb_get_cpu_count() * 3 / 2 );
    param.i_width      = job->width;
    param.i_height     = job->height;
    param.i_fps_num    = job->vrate;
    param.i_fps_den    = job->vrate_base;
    param.i_keyint_max = 20 * job->vrate / job->vrate_base;
    param.i_log_level  = X264_LOG_INFO;
    if( job->h264_level )
    {
        param.b_cabac     = 0;
        param.i_level_idc = job->h264_level;
        hb_log( "encx264: encoding at level %i",
                param.i_level_idc );
    }

    /* Slightly faster with minimal quality lost */
    param.analyse.i_subpel_refine = 4;

    /*
       	This section passes the string x264opts to libx264 for parsing into 
        parameter names and values.

        The string is set up like this:
        option1=value1:option2=value 2

        So, you have to iterate through based on the colons, and then put 
        the left side of the equals sign in "name" and the right side into
        "value." Then you hand those strings off to x264 for interpretation.

        This is all based on the universal x264 option handling Loren
        Merritt implemented in the Mplayer/Mencoder project.
     */

    char *x264opts = job->x264opts;
    if( x264opts != NULL && *x264opts != '\0' )
    {
        while( *x264opts )
        {
            char *name = x264opts;
            char *value;
            int ret;

            x264opts += strcspn( x264opts, ":" );
            if( *x264opts )
            {
                *x264opts = 0;
                x264opts++;
            }

            value = strchr( name, '=' );
            if( value )
            {
                *value = 0;
                value++;
            }

            /*
               When B-frames are enabled, the max frame count increments
               by 1 (regardless of the number of B-frames). If you don't
               change the duration of the video track when you mux, libmp4
               barfs.  So, check if the x264opts are using B-frames, and
               when they are, set the boolean job->areBframes as true.
             */

            if( !( strcmp( name, "bframes" ) ) )
            {
                if( atoi( value ) > 0 )
                {
                    job->areBframes = 1;
                }
            }

            /* Note b-pyramid here, so the initial delay can be doubled */
            if( !( strcmp( name, "b-pyramid" ) ) )
            {
                if( value != NULL )
                {
                    if( atoi( value ) > 0 )
                    {
                        job->areBframes = 2;
                    }
                }
                else
                {
                    job->areBframes = 2;
                }
            }

            /* Here's where the strings are passed to libx264 for parsing. */
            ret = x264_param_parse( &param, name, value );

            /* 	Let x264 sanity check the options for us*/
            if( ret == X264_PARAM_BAD_NAME )
                hb_log( "x264 options: Unknown suboption %s", name );
            if( ret == X264_PARAM_BAD_VALUE )
                hb_log( "x264 options: Bad argument %s=%s", name, value ? value : "(null)" );
        }
    }


    if( job->pixel_ratio )
    {
        param.vui.i_sar_width = job->pixel_aspect_width;
        param.vui.i_sar_height = job->pixel_aspect_height;

        hb_log( "encx264: encoding with stored aspect %d/%d",
                param.vui.i_sar_width, param.vui.i_sar_height );
    }


    if( job->vquality >= 0.0 && job->vquality <= 1.0 )
    {
        switch( job->crf )
        {
            case 1:
                /*Constant RF*/
                param.rc.i_rc_method = X264_RC_CRF;
                param.rc.f_rf_constant = 51 - job->vquality * 51;
                hb_log( "encx264: Encoding at constant RF %f",
                        param.rc.f_rf_constant );
                break;

            case 0:
                /*Constant QP*/
                param.rc.i_rc_method = X264_RC_CQP;
                param.rc.i_qp_constant = 51 - job->vquality * 51;
                hb_log( "encx264: encoding at constant QP %d",
                        param.rc.i_qp_constant );
                break;
        }
    }
    else
    {
        /* Rate control */
        param.rc.i_rc_method = X264_RC_ABR;
        param.rc.i_bitrate = job->vbitrate;
        switch( job->pass )
        {
            case 1:
                param.rc.b_stat_write  = 1;
                param.rc.psz_stat_out = pv->filename;
                break;
            case 2:
                param.rc.b_stat_read = 1;
                param.rc.psz_stat_in = pv->filename;
