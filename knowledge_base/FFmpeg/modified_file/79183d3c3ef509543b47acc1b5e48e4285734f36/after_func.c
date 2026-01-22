 *
 */

#include <stdio.h>
#include <stdlib.h>

#include "avcodec.h"
#include "bytestream.h"
#include "lcl.h"
#include "libavutil/lzo.h"

#if CONFIG_ZLIB_DECODER
#include <zlib.h>
#endif

/*
 * Decoder context
 */
typedef struct LclDecContext {
    AVFrame pic;

    // Image type
    int imgtype;
    // Compression type
    int compression;
    // Flags
    int flags;
    // Decompressed data size
    unsigned int decomp_size;
    // Decompression buffer
    unsigned char* decomp_buf;
#if CONFIG_ZLIB_DECODER
    z_stream zstream;
#endif
} LclDecContext;


/**
 * \param srcptr compressed source buffer, must be padded with at least 5 extra bytes
