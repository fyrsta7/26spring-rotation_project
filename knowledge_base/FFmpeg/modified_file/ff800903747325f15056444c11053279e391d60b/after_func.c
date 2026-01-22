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
