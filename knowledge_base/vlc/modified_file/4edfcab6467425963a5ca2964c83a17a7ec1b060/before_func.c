    vlc_mutex_destroy(&vd->sys->lock);

    Direct3D9Close(vd);

    CommonClean(vd);

    Direct3D9Destroy(vd);

    free(vd->sys);
}

static void DestroyPicture(picture_t *picture)
{
    IDirect3DSurface9_Release(picture->p_sys->surface);

    free(picture->p_sys);
    free(picture);
}

/**
 * It locks the surface associated to the picture and get the surface
 * descriptor which amongst other things has the pointer to the picture
 * data and its pitch.
 */
static int Direct3D9LockSurface(picture_t *picture)
{
    /* Lock the surface to get a valid pointer to the picture buffer */
    D3DLOCKED_RECT d3drect;
    HRESULT hr = IDirect3DSurface9_LockRect(picture->p_sys->surface, &d3drect, NULL, 0);
    if (FAILED(hr)) {
        return VLC_EGENERIC;
    }

    CommonUpdatePicture(picture, NULL, d3drect.pBits, d3drect.Pitch);
    return VLC_SUCCESS;
}
/**
 * It unlocks the surface associated to the picture.
 */
static void Direct3D9UnlockSurface(picture_t *picture)
{
    /* Unlock the Surface */
    HRESULT hr = IDirect3DSurface9_UnlockRect(picture->p_sys->surface);
    if (FAILED(hr)) {
        //msg_Dbg(vd, "Failed IDirect3DSurface9_UnlockRect: 0x%0lx", hr);
    }
}

/* */
static picture_pool_t *Direct3D9CreatePicturePool(vout_display_t *vd, unsigned count)
{
    picture_t**       pictures = NULL;
    unsigned          picture_count = 0;

    pictures = calloc(count, sizeof(*pictures));
    if (!pictures)
        goto error;

    D3DFORMAT format;
    switch (vd->fmt.i_chroma)
    {
    case VLC_CODEC_D3D9_OPAQUE_10B:
        format = MAKEFOURCC('P','0','1','0');
        break;
    case VLC_CODEC_D3D9_OPAQUE:
        format = MAKEFOURCC('N','V','1','2');
        break;
    default:
        format = vd->sys->d3dtexture_format->format;
        break;
    }

    for (picture_count = 0; picture_count < count; ++picture_count)
    {
        picture_sys_t *picsys = malloc(sizeof(*picsys));
        if (unlikely(picsys == NULL))
