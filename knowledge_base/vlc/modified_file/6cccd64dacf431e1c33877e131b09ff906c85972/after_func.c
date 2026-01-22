}

/* subtitle_ParseSubRipTiming
 * Parses SubRip timing.
 */
static int subtitle_ParseSubRipTiming( subtitle_t *p_subtitle,
                                       const char *s )
{
    int i_result = VLC_EGENERIC;
    char *psz_start, *psz_stop;
    psz_start = malloc( strlen(s) + 1 );
    psz_stop = malloc( strlen(s) + 1 );

    if( sscanf( s, "%s --> %s", psz_start, psz_stop) == 2 &&
        subtitle_ParseSubRipTimingValue( &p_subtitle->i_start, psz_start ) == VLC_SUCCESS &&
        subtitle_ParseSubRipTimingValue( &p_subtitle->i_stop,  psz_stop ) == VLC_SUCCESS )
    {
        i_result = VLC_SUCCESS;
    }

    free(psz_start);
    free(psz_stop);

    return i_result;
}
/* ParseSubRip
 */
static int  ParseSubRip( demux_t *p_demux, subtitle_t *p_subtitle,
                         int i_idx )
{
    VLC_UNUSED( i_idx );
    return ParseSubRipSubViewer( p_demux, p_subtitle,
                                 &subtitle_ParseSubRipTiming,
                                 false );
}

/* subtitle_ParseSubViewerTiming
 * Parses SubViewer timing.
 */
static int subtitle_ParseSubViewerTiming( subtitle_t *p_subtitle,
                                   const char *s )
{
    int h1, m1, s1, d1, h2, m2, s2, d2;

    if( sscanf( s, "%d:%d:%d.%d,%d:%d:%d.%d",
                &h1, &m1, &s1, &d1, &h2, &m2, &s2, &d2) == 8 )
    {
        p_subtitle->i_start = ( (int64_t)h1 * 3600*1000 +
                                (int64_t)m1 * 60*1000 +
                                (int64_t)s1 * 1000 +
                                (int64_t)d1 ) * 1000;

        p_subtitle->i_stop  = ( (int64_t)h2 * 3600*1000 +
                                (int64_t)m2 * 60*1000 +
                                (int64_t)s2 * 1000 +
                                (int64_t)d2 ) * 1000;
        return VLC_SUCCESS;
    }
    return VLC_EGENERIC;
}

/* ParseSubViewer
 */
static int  ParseSubViewer( demux_t *p_demux, subtitle_t *p_subtitle,
                            int i_idx )
{
    VLC_UNUSED( i_idx );

    return ParseSubRipSubViewer( p_demux, p_subtitle,
                                 &subtitle_ParseSubViewerTiming,
                                 true );
}

/* ParseSSA
 */
static int  ParseSSA( demux_t *p_demux, subtitle_t *p_subtitle,
                      int i_idx )
{
