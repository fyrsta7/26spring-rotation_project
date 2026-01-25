/*****************************************************************************
 * Handshake : Init audioscrobbler connection
 *****************************************************************************/
static int Handshake(intf_thread_t *p_this)
{
    char                *psz_username, *psz_password;
    char                *psz_scrobbler_url;
    time_t              timestamp;
    char                psz_timestamp[21];

    vlc_hash_md5_t      struct_md5;

    stream_t            *p_stream;
    char                *psz_handshake_url;
    uint8_t             p_buffer[1024];
    char                *p_buffer_pos;

    int                 i_ret;
    char                *psz_url;

    intf_thread_t       *p_intf                 = (intf_thread_t*) p_this;
    intf_sys_t          *p_sys                  = p_this->p_sys;

    psz_username = var_InheritString(p_this, "lastfm-username");
    psz_password = var_InheritString(p_this, "lastfm-password");

    /* username or password have not been setup */
    if (EMPTY_STR(psz_username) || EMPTY_STR(psz_password))
    {
        free(psz_username);
        free(psz_password);
        return VLC_EINVAL;
    }

    time(&timestamp);

    /* generates a md5 hash of the password */
    vlc_hash_md5_Init(&struct_md5);
    vlc_hash_md5_Update(&struct_md5, psz_password, strlen(psz_password));
    free(psz_password);

    char psz_password_md5[VLC_HASH_MD5_DIGEST_HEX_SIZE];
    vlc_hash_FinishHex(&struct_md5, psz_password_md5);

    snprintf(psz_timestamp, sizeof(psz_timestamp), "%"PRIu64,
              (uint64_t)timestamp);

    /* generates a md5 hash of :
     * - md5 hash of the password, plus
     * - timestamp in clear text
     */
    vlc_hash_md5_Init(&struct_md5);
    vlc_hash_md5_Update(&struct_md5, psz_password_md5, sizeof(psz_password_md5) - 1);
    vlc_hash_md5_Update(&struct_md5, psz_timestamp, strlen(psz_timestamp));
    char psz_auth_token[VLC_HASH_MD5_DIGEST_HEX_SIZE];
    vlc_hash_FinishHex(&struct_md5, psz_auth_token);

    psz_scrobbler_url = var_InheritString(p_this, "scrobbler-url");
    if (!psz_scrobbler_url)
    {
        free(psz_username);
        return VLC_ENOMEM;
    }

    i_ret = asprintf(&psz_handshake_url,
    "http://%s/?hs=true&p=1.2&c="CLIENT_NAME"&v="CLIENT_VERSION"&u=%s&t=%s&a=%s"
    , psz_scrobbler_url, psz_username, psz_timestamp, psz_auth_token);

    free(psz_scrobbler_url);
    free(psz_username);
    if (i_ret == -1)
        return VLC_ENOMEM;

    /* send the http handshake request */
    p_stream = vlc_stream_NewURL(p_intf, psz_handshake_url);
    free(psz_handshake_url);

    if (!p_stream)
        return VLC_EGENERIC;

    /* read answer */
    i_ret = vlc_stream_Read(p_stream, p_buffer, sizeof(p_buffer) - 1);
    if (i_ret <= 0)
    {
        vlc_stream_Delete(p_stream);
        return VLC_EGENERIC;
    }
    p_buffer[i_ret] = '\0';
    vlc_stream_Delete(p_stream);

    p_buffer_pos = strstr((char*) p_buffer, "FAILED ");
    if (p_buffer_pos)
    {
        /* handshake request failed, sorry */
        msg_Err(p_this, "last.fm handshake failed: %s", p_buffer_pos + 7);
        return VLC_EGENERIC;
    }

    if (strstr((char*) p_buffer, "BADAUTH"))
    {
        /* authentication failed, bad username/password combination */
        vlc_dialog_display_error(p_this,
            _("last.fm: Authentication failed"),
            "%s", _("last.fm username or password is incorrect. "
              "Please verify your settings and relaunch VLC."));
        return VLC_AUDIOSCROBBLER_EFATAL;
    }

    if (strstr((char*) p_buffer, "BANNED"))
    {
        /* oops, our version of vlc has been banned by last.fm servers */
        msg_Err(p_intf, "This version of VLC has been banned by last.fm. "
                         "You should upgrade VLC, or disable the last.fm plugin.");
        return VLC_AUDIOSCROBBLER_EFATAL;
    }

    if (strstr((char*) p_buffer, "BADTIME"))
    {
        /* The system clock isn't good */
        msg_Err(p_intf, "last.fm handshake failed because your clock is too "
                         "much shifted. Please correct it, and relaunch VLC.");
        return VLC_AUDIOSCROBBLER_EFATAL;
    }

    p_buffer_pos = strstr((char*) p_buffer, "OK");
    if (!p_buffer_pos)
        goto proto;

    p_buffer_pos = strstr(p_buffer_pos, "\n");
    if (!p_buffer_pos || strlen(p_buffer_pos) < 33)
        goto proto;
    p_buffer_pos++; /* we skip the '\n' */

    /* save the session ID */
    memcpy(p_sys->psz_auth_token, p_buffer_pos, 32);
    p_sys->psz_auth_token[32] = '\0';

    p_buffer_pos = strstr(p_buffer_pos, "http://");
    if (!p_buffer_pos || strlen(p_buffer_pos) == 7)
        goto proto;

    /* We need to read the nowplaying url */
    psz_url = strndup(p_buffer_pos, strcspn(p_buffer_pos, "\n"));
    if (!psz_url)
        goto oom;

    vlc_UrlParse(&p_sys->p_nowp_url, psz_url);
    free(psz_url);
    if (p_sys->p_nowp_url.psz_host == NULL ||
        p_sys->p_nowp_url.i_port == 0)
    {
        vlc_UrlClean(&p_sys->p_nowp_url);
        goto proto;
    }
    p_buffer_pos += strcspn(p_buffer_pos, "\n");

    p_buffer_pos = strstr(p_buffer_pos, "http://");
    if (!p_buffer_pos || strlen(p_buffer_pos) == 7)
        goto proto;

    /* We need to read the submission url */
    psz_url = strndup(p_buffer_pos, strcspn(p_buffer_pos, "\n"));
    if (!psz_url)
        goto oom;

    /* parse the submission url */
    vlc_UrlParse(&p_sys->p_submit_url, psz_url);
    free(psz_url);
    if (p_sys->p_submit_url.psz_host == NULL ||
        p_sys->p_submit_url.i_port == 0)
    {
        vlc_UrlClean(&p_sys->p_nowp_url);
        vlc_UrlClean(&p_sys->p_submit_url);
        goto proto;
    }

    return VLC_SUCCESS;

oom:
    return VLC_ENOMEM;

proto:
    msg_Err(p_intf, "Handshake: can't recognize server protocol");
    return VLC_EGENERIC;
}