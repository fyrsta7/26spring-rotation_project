    SpatiallyCoded,
    EntropyCoded,
};

static ErrorOr<NonnullRefPtr<Bitmap>> decode_webp_chunk_VP8L_image(WebPLoadingContext& context, ImageKind image_kind, BitmapFormat format, IntSize const& size, LittleEndianInputBitStream& bit_stream)
{
    // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#623_decoding_entropy-coded_image_data
    // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#523_color_cache_coding
    // spatially-coded-image =  color-cache-info meta-prefix data
    // entropy-coded-image   =  color-cache-info data

    // color-cache-info      =  %b0
    // color-cache-info      =/ (%b1 4BIT) ; 1 followed by color cache size
    bool has_color_cache_info = TRY(bit_stream.read_bits(1));
    u16 color_cache_size = 0;
    u8 color_cache_code_bits;
    dbgln_if(WEBP_DEBUG, "has_color_cache_info {}", has_color_cache_info);
    Vector<ARGB32, 32> color_cache;
    if (has_color_cache_info) {
        color_cache_code_bits = TRY(bit_stream.read_bits(4));

        // "The range of allowed values for color_cache_code_bits is [1..11]. Compliant decoders must indicate a corrupted bitstream for other values."
        if (color_cache_code_bits < 1 || color_cache_code_bits > 11)
            return context.error("WebPImageDecoderPlugin: VP8L invalid color_cache_code_bits");

        color_cache_size = 1 << color_cache_code_bits;
        dbgln_if(WEBP_DEBUG, "color_cache_size {}", color_cache_size);

        TRY(color_cache.try_resize(color_cache_size));
    }

    int num_prefix_groups = 1;
    RefPtr<Gfx::Bitmap> entropy_image;
    int prefix_bits = 0;
    if (image_kind == ImageKind::SpatiallyCoded) {
        // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#622_decoding_of_meta_prefix_codes
        // In particular, the "Entropy image" subsection.
        // "Meta prefix codes may be used only when the image is being used in the role of an ARGB image."
        // meta-prefix           =  %b0 / (%b1 entropy-image)
        bool has_meta_prefix = TRY(bit_stream.read_bits(1));
        dbgln_if(WEBP_DEBUG, "has_meta_prefix {}", has_meta_prefix);
        if (has_meta_prefix) {
            prefix_bits = TRY(bit_stream.read_bits(3)) + 2;
            dbgln_if(WEBP_DEBUG, "prefix_bits {}", prefix_bits);
            int block_size = 1 << prefix_bits;
            IntSize prefix_size { ceil_div(size.width(), block_size), ceil_div(size.height(), block_size) };

            entropy_image = TRY(decode_webp_chunk_VP8L_image(context, ImageKind::EntropyCoded, BitmapFormat::BGRx8888, prefix_size, bit_stream));

            // A "meta prefix image" or "entropy image" can tell the decoder to use different PrefixCodeGroup for
            // tiles of the main, spatially coded, image. It's a bit hidden in the spec:
            //      "The red and green components of a pixel define the meta prefix code used in a particular block of the ARGB image."
            //      ...
            //      "The number of prefix code groups in the ARGB image can be obtained by finding the largest meta prefix code from the entropy image"
            // That is, if a meta prefix image is present, the main image has more than one PrefixCodeGroup,
            // and the highest value in the meta prefix image determines how many exactly.
            u16 largest_meta_prefix_code = 0;
            for (ARGB32& pixel : *entropy_image) {
                u16 meta_prefix_code = (pixel >> 8) & 0xffff;
                if (meta_prefix_code > largest_meta_prefix_code)
                    largest_meta_prefix_code = meta_prefix_code;
            }
            dbgln_if(WEBP_DEBUG, "largest meta prefix code {}", largest_meta_prefix_code);

            num_prefix_groups = largest_meta_prefix_code + 1;
        }
    }

    // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#52_encoding_of_image_data
    // "The encoded image data consists of several parts:
    //    1. Decoding and building the prefix codes [AMENDED2]
    //    2. Meta prefix codes
    //    3. Entropy-coded image data"
    // data                  =  prefix-codes lz77-coded-image
    // prefix-codes          =  prefix-code-group *prefix-codes

    Vector<PrefixCodeGroup, 1> groups;
    for (int i = 0; i < num_prefix_groups; ++i)
        TRY(groups.try_append(TRY(decode_webp_chunk_VP8L_prefix_code_group(context, color_cache_size, bit_stream))));

    auto bitmap = TRY(Bitmap::create(format, size));

    // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#522_lz77_backward_reference
    struct Offset {
        i8 x, y;
    };
    // clang-format off
    Array<Offset, 120> distance_map { {
        {0, 1}, {1, 0},
        {1, 1}, {-1, 1}, {0, 2}, { 2, 0},
        {1, 2}, {-1, 2}, {2, 1}, {-2, 1},
        {2, 2}, {-2, 2}, {0, 3}, { 3, 0}, { 1, 3}, {-1, 3}, { 3, 1}, {-3, 1},
        {2, 3}, {-2, 3}, {3, 2}, {-3, 2}, { 0, 4}, { 4, 0}, { 1, 4}, {-1, 4}, { 4, 1}, {-4, 1},
        {3, 3}, {-3, 3}, {2, 4}, {-2, 4}, { 4, 2}, {-4, 2}, { 0, 5},
        {3, 4}, {-3, 4}, {4, 3}, {-4, 3}, { 5, 0}, { 1, 5}, {-1, 5}, { 5, 1}, {-5, 1}, { 2, 5}, {-2, 5}, { 5, 2}, {-5, 2},
        {4, 4}, {-4, 4}, {3, 5}, {-3, 5}, { 5, 3}, {-5, 3}, { 0, 6}, { 6, 0}, { 1, 6}, {-1, 6}, { 6, 1}, {-6, 1}, { 2, 6}, {-2, 6}, {6, 2}, {-6, 2},
        {4, 5}, {-4, 5}, {5, 4}, {-5, 4}, { 3, 6}, {-3, 6}, { 6, 3}, {-6, 3}, { 0, 7}, { 7, 0}, { 1, 7}, {-1, 7},
        {5, 5}, {-5, 5}, {7, 1}, {-7, 1}, { 4, 6}, {-4, 6}, { 6, 4}, {-6, 4}, { 2, 7}, {-2, 7}, { 7, 2}, {-7, 2}, { 3, 7}, {-3, 7}, {7, 3}, {-7, 3},
        {5, 6}, {-5, 6}, {6, 5}, {-6, 5}, { 8, 0}, { 4, 7}, {-4, 7}, { 7, 4}, {-7, 4}, { 8, 1}, { 8, 2},
        {6, 6}, {-6, 6}, {8, 3}, { 5, 7}, {-5, 7}, { 7, 5}, {-7, 5}, { 8, 4},
        {6, 7}, {-6, 7}, {7, 6}, {-7, 6}, { 8, 5},
        {7, 7}, {-7, 7}, {8, 6},
        {8, 7},
    } };
    // clang-format on

    // lz77-coded-image      =
    //     *((argb-pixel / lz77-copy / color-cache-code) lz77-coded-image)
    // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#623_decoding_entropy-coded_image_data
    ARGB32* begin = bitmap->begin();
    ARGB32* end = bitmap->end();
    ARGB32* pixel = begin;

    auto prefix_group = [prefix_bits, begin, &groups, size, &entropy_image](ARGB32* pixel) {
        if (!prefix_bits)
            return groups[0];

        size_t offset = pixel - begin;
        int x = offset % size.width();
        int y = offset / size.width();

        int meta_prefix_code = (entropy_image->scanline(y >> prefix_bits)[x >> prefix_bits] >> 8) & 0xffff;
        return groups[meta_prefix_code];
    };

    auto emit_pixel = [&pixel, &color_cache, color_cache_size, color_cache_code_bits](ARGB32 color) {
        // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#523_color_cache_coding
        // "The state of the color cache is maintained by inserting every pixel, be it produced by backward referencing or as literals, into the cache in the order they appear in the stream."
        *pixel++ = color;
        if (color_cache_size)
            color_cache[(0x1e35a7bd * color) >> (32 - color_cache_code_bits)] = color;
    };

    while (pixel < end) {
        auto const& group = prefix_group(pixel);

        auto symbol = TRY(group[0].read_symbol(bit_stream));
        if (symbol >= 256u + 24u + color_cache_size)
            return context.error("WebPImageDecoderPlugin: Symbol out of bounds");

        // "1. if S < 256"
        if (symbol < 256u) {
            // "a. Use S as the green component."
            u8 g = symbol;

            // "b. Read red from the bitstream using prefix code #2."
            u8 r = TRY(group[1].read_symbol(bit_stream));

            // "c. Read blue from the bitstream using prefix code #3."
            u8 b = TRY(group[2].read_symbol(bit_stream));

            // "d. Read alpha from the bitstream using prefix code #4."
            u8 a = TRY(group[3].read_symbol(bit_stream));

            emit_pixel(Color(r, g, b, a).value());
        }
        // "2. if S >= 256 && S < 256 + 24"
        else if (symbol < 256u + 24u) {
            auto prefix_value = [&bit_stream](u8 prefix_code) -> ErrorOr<u32> {
                // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#522_lz77_backward_reference
                if (prefix_code < 4)
                    return prefix_code + 1;
                int extra_bits = (prefix_code - 2) >> 1;
                int offset = (2 + (prefix_code & 1)) << extra_bits;
                return offset + TRY(bit_stream.read_bits(extra_bits)) + 1;
            };

            // "a. Use S - 256 as a length prefix code."
            u8 length_prefix_code = symbol - 256;

            // "b. Read extra bits for length from the bitstream."
            // "c. Determine backward-reference length L from length prefix code and the extra bits read."
            u32 length = TRY(prefix_value(length_prefix_code));

            // "d. Read distance prefix code from the bitstream using prefix code #5."
            u8 distance_prefix_code = TRY(group[4].read_symbol(bit_stream));

            // "e. Read extra bits for distance from the bitstream."
            // "f. Determine backward-reference distance D from distance prefix code and the extra bits read."
            i32 distance = TRY(prefix_value(distance_prefix_code));

            // "g. Copy the L pixels (in scan-line order) from the sequence of pixels prior to them by D pixels."

            // https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification#522_lz77_backward_reference
            // "Distance codes larger than 120 denote the pixel-distance in scan-line order, offset by 120."
            // "The smallest distance codes [1..120] are special, and are reserved for a close neighborhood of the current pixel."
            if (distance <= 120) {
                auto offset = distance_map[distance - 1];
                distance = offset.x + offset.y * bitmap->physical_width();
                if (distance < 1)
                    distance = 1;
            } else {
                distance = distance - 120;
            }

            if (pixel - begin < distance) {
                dbgln_if(WEBP_DEBUG, "invalid backref, {} < {}", pixel - begin, distance);
                return context.error("WebPImageDecoderPlugin: Backward reference distance out of bounds");
            }

            if (end - pixel < length) {
                dbgln_if(WEBP_DEBUG, "invalid length, {} < {}", end - pixel, length);
                return context.error("WebPImageDecoderPlugin: Backward reference length out of bounds");
            }

            ARGB32* src = pixel - distance;
            for (u32 i = 0; i < length; ++i)
                emit_pixel(src[i]);
        }
        // "3. if S >= 256 + 24"
        else {
            // "a. Use S - (256 + 24) as the index into the color cache."
            unsigned index = symbol - (256 + 24);

            // "b. Get ARGB color from the color cache at that index."
            if (index >= color_cache_size)
                return context.error("WebPImageDecoderPlugin: Color cache index out of bounds");
            *pixel++ = color_cache[index];
        }
