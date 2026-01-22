#include "sws_utils.h"
#include "fmt-conversion.h"

const struct m_opt_choice_alternatives mp_spherical_names[] = {
    {"auto",        MP_SPHERICAL_AUTO},
    {"none",        MP_SPHERICAL_NONE},
    {"unknown",     MP_SPHERICAL_UNKNOWN},
    {"equirect",    MP_SPHERICAL_EQUIRECTANGULAR},
    {0}
};

// Determine strides, plane sizes, and total required size for an image
// allocation. Returns total size on success, <0 on error. Unused planes
// have out_stride/out_plane_size to 0, and out_plane_offset set to -1 up
// until MP_MAX_PLANES-1.
static int mp_image_layout(int imgfmt, int w, int h, int stride_align,
                           int out_stride[MP_MAX_PLANES],
                           int out_plane_offset[MP_MAX_PLANES],
                           int out_plane_size[MP_MAX_PLANES])
{
    struct mp_imgfmt_desc desc = mp_imgfmt_get_desc(imgfmt);
    struct mp_image_params params = {.imgfmt = imgfmt, .w = w, .h = h};

    if (!mp_image_params_valid(&params) || desc.flags & MP_IMGFLAG_HWACCEL)
        return -1;

    // Note: for non-mod-2 4:2:0 YUV frames, we have to allocate an additional
    //       top/right border. This is needed for correct handling of such
    //       images in filter and VO code (e.g. vo_vdpau or vo_gpu).

    for (int n = 0; n < MP_MAX_PLANES; n++) {
        int alloc_w = mp_chroma_div_up(w, desc.xs[n]);
        int alloc_h = MP_ALIGN_UP(h, 32) >> desc.ys[n];
        int line_bytes = (alloc_w * desc.bpp[n] + 7) / 8;
        out_stride[n] = MP_ALIGN_UP(line_bytes, stride_align);
        // also align to a multiple of desc.bytes[n]
