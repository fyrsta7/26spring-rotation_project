    assert(sys->smb2 == smb2); \
    if (VLC_SMB2_CHECK_STATUS(access, status)) \
        return

static void
smb2_generic_cb(struct smb2_context *smb2, int status, void *data,
                void *private_data)
{
    VLC_UNUSED(data);
    VLC_SMB2_GENERIC_CB();
}

static void
smb2_read_cb(struct smb2_context *smb2, int status, void *data,
             void *private_data)
{
    VLC_UNUSED(data);
    VLC_SMB2_GENERIC_CB();

    if (status == 0)
        sys->eof = true;
    else
        sys->res.read.len = status;
}
