    p_credential->p_url = p_url;
}

void
vlc_credential_clean(vlc_credential *p_credential)
{
    if (p_credential->i_entries_count > 0)
        vlc_keystore_release_entries(p_credential->p_entries,
                                     p_credential->i_entries_count);
    if (p_credential->p_keystore)
        vlc_keystore_release(p_credential->p_keystore);

    free(p_credential->psz_split_domain);
    free(p_credential->psz_var_username);
    free(p_credential->psz_var_password);
    free(p_credential->psz_dialog_username);
    free(p_credential->psz_dialog_password);
}

#undef vlc_credential_get
bool
vlc_credential_get(vlc_credential *p_credential, vlc_object_t *p_parent,
                   const char *psz_option_username,
                   const char *psz_option_password,
                   const char *psz_dialog_title,
                   const char *psz_dialog_fmt, ...)
{
    assert(p_credential && p_parent);
    const vlc_url_t *p_url = p_credential->p_url;

    if (!is_url_valid(p_url))
    {
        msg_Err(p_parent, "vlc_credential_get: invalid url");
        return false;
    }

    p_credential->b_from_keystore = false;
    /* Don't set username to NULL, we may want to use the last one set */
    p_credential->psz_password = NULL;

    while (!is_credential_valid(p_credential))
    {
        /* First, fetch credential from URL (if any).
         * Secondly, fetch credential from VLC Options (if any).
         * Thirdly, fetch credential from keystore (if any) using user and realm
         * previously set by the caller, the URL or by VLC Options.
         * Finally, fetch credential from the dialog (if any). This last will be
         * repeated until user cancel the dialog. */

        switch (p_credential->i_get_order)
        {
        case GET_FROM_URL:
            p_credential->psz_username = p_url->psz_username;
            p_credential->psz_password = p_url->psz_password;

            if (p_credential->psz_password)
                msg_Warn(p_parent, "Password in a URI is DEPRECATED");

            if (p_url->psz_username && protocol_is_smb(p_url))
                smb_split_domain(p_credential);
            p_credential->i_get_order++;
            break;

        case GET_FROM_OPTION:
            free(p_credential->psz_var_username);
            free(p_credential->psz_var_password);
            p_credential->psz_var_username =
            p_credential->psz_var_password = NULL;

            if (psz_option_username)
                p_credential->psz_var_username =
                    var_InheritString(p_parent, psz_option_username);
            if (psz_option_password)
                p_credential->psz_var_password =
                    var_InheritString(p_parent, psz_option_password);

            if (p_credential->psz_var_username)
                p_credential->psz_username = p_credential->psz_var_username;
            if (p_credential->psz_var_password)
                p_credential->psz_password = p_credential->psz_var_password;

            p_credential->i_get_order++;
            break;

        case GET_FROM_MEMORY_KEYSTORE:
        {
            if (!psz_dialog_title || !psz_dialog_fmt)
                return false;

            vlc_keystore *p_keystore = get_memory_keystore(p_parent);
            if (p_keystore != NULL)
                credential_find_keystore(p_credential, p_keystore);
            p_credential->i_get_order++;
            break;
        }

        case GET_FROM_KEYSTORE:
            if (!psz_dialog_title || !psz_dialog_fmt)
                return false;

            if (p_credential->p_keystore == NULL)
                p_credential->p_keystore = vlc_keystore_create(p_parent);
            if (p_credential->p_keystore != NULL)
                credential_find_keystore(p_credential, p_credential->p_keystore);

            p_credential->i_get_order++;
            break;

        default:
        case GET_FROM_DIALOG:
            if (!psz_dialog_title || !psz_dialog_fmt)
                return false;
            char *psz_dialog_username = NULL;
            char *psz_dialog_password = NULL;
            va_list ap;
            va_start(ap, psz_dialog_fmt);
            bool *p_store = p_credential->p_keystore != NULL ?
                            &p_credential->b_store : NULL;
            int i_ret =
                vlc_dialog_wait_login_va(p_parent,
                                         &psz_dialog_username,
                                         &psz_dialog_password, p_store,
                                         p_credential->psz_username,
                                         psz_dialog_title, psz_dialog_fmt, ap);
