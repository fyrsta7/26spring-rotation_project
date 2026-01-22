
    var_AddCallback( p_filter, "overlay-input", AdjustCallback, p_sys );
    var_AddCallback( p_filter, "overlay-output", AdjustCallback, p_sys );

    RegisterCommand( p_filter );
    return VLC_SUCCESS;
}

/*****************************************************************************
 * Destroy: destroy adjust video thread output method
 *****************************************************************************
 * Terminate an output method created by adjustCreateOutputMethod
 *****************************************************************************/
static void Destroy( filter_t *p_filter )
{
    filter_sys_t *p_sys = p_filter->p_sys;

    BufferDestroy( &p_sys->input );
    BufferDestroy( &p_sys->output );
    QueueDestroy( &p_sys->atomic );
    QueueDestroy( &p_sys->pending );
    QueueDestroy( &p_sys->processed );
    do_ListDestroy( &p_sys->overlays );
    UnregisterCommand( p_filter );

    var_DelCallback( p_filter, "overlay-input", AdjustCallback, p_sys );
    var_DelCallback( p_filter, "overlay-output", AdjustCallback, p_sys );

    free( p_sys->psz_inputfile );
    free( p_sys->psz_outputfile );
    free( p_sys );
}

/*****************************************************************************
 * Render: displays previously rendered output
 *****************************************************************************
 * This function send the currently rendered image to adjust modified image,
 * waits until it is displayed and switch the two rendering buffers, preparing
 * next frame.
 *****************************************************************************/
static subpicture_t *Filter( filter_t *p_filter, vlc_tick_t date )
{
    filter_sys_t *p_sys = p_filter->p_sys;

    /* We might need to open these at any time. */
    vlc_mutex_lock( &p_sys->lock );
    if( p_sys->i_inputfd == -1 )
    {
        p_sys->i_inputfd = vlc_open( p_sys->psz_inputfile, O_RDONLY | O_NONBLOCK );
        if( p_sys->i_inputfd == -1 )
        {
            msg_Warn( p_filter, "Failed to grab input file: %s (%s)",
                      p_sys->psz_inputfile, vlc_strerror_c(errno) );
        }
        else
        {
            msg_Info( p_filter, "Grabbed input file: %s",
                      p_sys->psz_inputfile );
        }
    }

    if( p_sys->i_outputfd == -1 )
    {
        p_sys->i_outputfd = vlc_open( p_sys->psz_outputfile,
                                  O_WRONLY | O_NONBLOCK );
        if( p_sys->i_outputfd == -1 )
        {
            if( errno != ENXIO )
            {
                msg_Warn( p_filter, "Failed to grab output file: %s (%s)",
                          p_sys->psz_outputfile, vlc_strerror_c(errno) );
            }
        }
        else
        {
            msg_Info( p_filter, "Grabbed output file: %s",
                      p_sys->psz_outputfile );
        }
    }
    vlc_mutex_unlock( &p_sys->lock );

    /* Read any waiting commands */
    if( p_sys->i_inputfd != -1 )
    {
        char p_buffer[1024];
        ssize_t i_len = read( p_sys->i_inputfd, p_buffer, 1024 );
        if( i_len == -1 )
        {
            /* We hit an error */
            if( errno != EAGAIN )
            {
                msg_Warn( p_filter, "Error on input file: %s",
                          vlc_strerror_c(errno) );
                vlc_close( p_sys->i_inputfd );
                p_sys->i_inputfd = -1;
            }
        }
        else if( i_len == 0 )
        {
            /* We hit the end-of-file */
        }
        else
        {
            BufferAdd( &p_sys->input, p_buffer, i_len );
        }
    }

    /* Parse any complete commands */
    char *p_end, *p_cmd;
    while( ( p_end = memchr( p_sys->input.p_begin, '\n',
                             p_sys->input.i_length ) ) )
    {
        commanddesc_t *p_cur = NULL;
        bool b_found = false;
        size_t i_index = 0;

        *p_end = '\0';
        p_cmd = BufferGetToken( &p_sys->input );

        msg_Info( p_filter, "Search command: %s", p_cmd );
        for( i_index = 0; i_index < p_sys->i_commands; i_index++ )
        {
            p_cur = p_sys->pp_commands[i_index];
            if( !strncmp( p_cur->psz_command, p_cmd, strlen(p_cur->psz_command) ) )
            {
                p_cmd[strlen(p_cur->psz_command)] = '\0';
                b_found = true;
                break;
            }
        }

        if( !b_found )
        {
            /* No matching command */
            msg_Err( p_filter, "Got invalid command: %s", p_cmd );
            BufferPrintf( &p_sys->output, "FAILURE: %d Invalid Command\n", VLC_EGENERIC );
        }
        else
        {
            msg_Info( p_filter, "Got valid command: %s", p_cmd );

            command_t *p_cmddesc = malloc( sizeof( command_t ) );
            if( !p_cmddesc )
                return NULL;

            p_cmd = p_cmd + strlen(p_cur->psz_command) +1;
            p_cmddesc->p_command = p_cur;
            p_cmddesc->p_command->pf_parser( p_cmd, p_end,
                                             &p_cmddesc->params );

            if( p_cmddesc->p_command->b_atomic && p_sys->b_atomic )
                QueueEnqueue( &p_sys->atomic, p_cmddesc );
            else
                QueueEnqueue( &p_sys->pending, p_cmddesc );
        }

        BufferDel( &p_sys->input, p_end - p_sys->input.p_begin + 1 );
    }

    /* Process any pending commands */
    command_t *p_command = NULL;
    while( (p_command = QueueDequeue( &p_sys->pending )) )
    {
        p_command->i_status =
            p_command->p_command->pf_execute( p_filter, &p_command->params,
                                              &p_command->results );
        QueueEnqueue( &p_sys->processed, p_command );
    }

    /* Output any processed commands */
    while( (p_command = QueueDequeue( &p_sys->processed )) )
    {
        if( p_command->i_status == VLC_SUCCESS )
        {
            const char *psz_success = "SUCCESS:";
            const char *psz_nl = "\n";
            BufferAdd( &p_sys->output, psz_success, 8 );
            p_command->p_command->pf_unparse( &p_command->results,
                                              &p_sys->output );
            BufferAdd( &p_sys->output, psz_nl, 1 );
        }
        else
        {
            BufferPrintf( &p_sys->output, "FAILURE: %d\n",
                          p_command->i_status );
        }
    }

    /* Try emptying the output buffer */
    if( p_sys->i_outputfd != -1 )
    {
        ssize_t i_len = vlc_write( p_sys->i_outputfd, p_sys->output.p_begin,
                                   p_sys->output.i_length );
        if( i_len == -1 )
        {
            /* We hit an error */
            if( errno != EAGAIN )
            {
                msg_Warn( p_filter, "Error on output file: %s",
                          vlc_strerror_c(errno) );
                vlc_close( p_sys->i_outputfd );
                p_sys->i_outputfd = -1;
            }
        }
        else
        {
            BufferDel( &p_sys->output, i_len );
        }
    }

    if( !p_sys->b_updated )
        return NULL;

    subpicture_t *p_spu = NULL;
    overlay_t *p_overlay = NULL;

    p_spu = filter_NewSubpicture( p_filter );
    if( !p_spu )
        return NULL;
