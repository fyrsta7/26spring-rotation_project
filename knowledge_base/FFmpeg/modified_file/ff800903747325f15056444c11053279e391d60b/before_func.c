#include "config.h"
#include "avutil.h"
#include "avassert.h"
#include "samplefmt.h"
#include "internal.h"

/**
 * @file
 * various utility functions
 */

#include "libavutil/ffversion.h"
const char av_util_ffversion[] = "FFmpeg version " FFMPEG_VERSION;

const char *av_version_info(void)
{
    return FFMPEG_VERSION;
}

unsigned avutil_version(void)
{
    static int checks_done;
    if (checks_done)
        return LIBAVUTIL_VERSION_INT;

    av_assert0(AV_SAMPLE_FMT_DBLP == 9);
