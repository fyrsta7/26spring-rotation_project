
error:
    *pp_message = p_message;
    free( buf );
    free( ppsz_command );
    return VLC_EGENERIC;
}

/*****************************************************************************
 * Media handling
 *****************************************************************************/
vlm_media_sys_t *vlm_MediaSearch( vlm_t *vlm, const char *psz_name )
{
    int i;

