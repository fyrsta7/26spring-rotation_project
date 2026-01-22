            vertex_order[0] = 1;
            vertex_order[1] = 2;
            vertex_order[2] = 3;
            vertex_order[3] = 0;
            break;
       default:
            vertex_order[0] = 0;
            vertex_order[1] = 1;
            vertex_order[2] = 2;
            vertex_order[3] = 3;
            break;
    }
}

static void Direct3D9SetupVertices(CUSTOMVERTEX *vertices,
                                  const RECT src_full,
                                  const RECT src_crop,
                                  const RECT dst,
                                  int alpha,
                                  video_orientation_t orientation)
{
    const float src_full_width  = src_full.right  - src_full.left;
    const float src_full_height = src_full.bottom - src_full.top;

    /* Vertices of the dst rectangle in the unrotated (clockwise) order. */
    const int vertices_coords[4][2] = {
        { dst.left,  dst.top    },
        { dst.right, dst.top    },
        { dst.right, dst.bottom },
        { dst.left,  dst.bottom },
    };

    /* Compute index remapping necessary to implement the rotation. */
    int vertex_order[4];
    orientationVertexOrder(orientation, vertex_order);

    for (int i = 0; i < 4; ++i) {
        vertices[i].x  = vertices_coords[vertex_order[i]][0];
        vertices[i].y  = vertices_coords[vertex_order[i]][1];
    }

    vertices[0].tu = src_crop.left / src_full_width;
    vertices[0].tv = src_crop.top  / src_full_height;

    vertices[1].tu = src_crop.right / src_full_width;
    vertices[1].tv = src_crop.top   / src_full_height;

    vertices[2].tu = src_crop.right  / src_full_width;
    vertices[2].tv = src_crop.bottom / src_full_height;

    vertices[3].tu = src_crop.left   / src_full_width;
    vertices[3].tv = src_crop.bottom / src_full_height;

    for (int i = 0; i < 4; i++) {
        /* -0.5f is a "feature" of DirectX and it seems to apply to Direct3d also */
        /* http://www.sjbrown.co.uk/2003/05/01/fix-directx-rasterisation/ */
        vertices[i].x -= 0.5;
        vertices[i].y -= 0.5;

        vertices[i].z       = 0.0f;
        vertices[i].rhw     = 1.0f;
        vertices[i].diffuse = D3DCOLOR_ARGB(alpha, 255, 255, 255);
    }
}

/**
 * It copies picture surface into a texture and setup the associated d3d_region_t.
 */
static int Direct3D9ImportPicture(vout_display_t *vd,
                                 d3d_region_t *region,
                                 LPDIRECT3DSURFACE9 source)
{
    vout_display_sys_t *sys = vd->sys;
    HRESULT hr;

    if (!source) {
        msg_Dbg(vd, "no surface to render?");
        return VLC_EGENERIC;
    }

    /* retrieve texture top-level surface */
    LPDIRECT3DSURFACE9 destination;
    hr = IDirect3DTexture9_GetSurfaceLevel(sys->d3dtex, 0, &destination);
    if (FAILED(hr)) {
        msg_Dbg(vd, "Failed IDirect3DTexture9_GetSurfaceLevel: 0x%0lx", hr);
        return VLC_EGENERIC;
    }

    /* Copy picture surface into texture surface
     * color space conversion happen here */
    RECT cropSource;
    cropSource.left = 0;
    cropSource.top = 0;
    cropSource.right = vd->fmt.i_visible_width;
    cropSource.bottom = vd->fmt.i_visible_height;
    hr = IDirect3DDevice9_StretchRect(sys->d3ddev, source, &cropSource, destination, NULL, D3DTEXF_LINEAR);
    IDirect3DSurface9_Release(destination);
    if (FAILED(hr)) {
        msg_Dbg(vd, "Failed IDirect3DDevice9_StretchRect: source 0x%p 0x%0lx", source, hr);
        return VLC_EGENERIC;
    }

    /* */
    region->texture = sys->d3dtex;
    Direct3D9SetupVertices(region->vertex,
                          vd->sys->rect_src,
                          vd->sys->rect_src_clipped,
                          vd->sys->rect_dest_clipped, 255, vd->fmt.orientation);
    return VLC_SUCCESS;
}

static void Direct3D9DeleteRegions(int count, d3d_region_t *region)
{
