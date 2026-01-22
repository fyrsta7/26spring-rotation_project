#if MICROPY_STREAMS_NON_BLOCK
#include <errno.h>
#if defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR)
#define EWOULDBLOCK 140
#endif
#endif

// This file defines generic Python stream read/write methods which
// dispatch to the underlying stream interface of an object.

// TODO: should be in mpconfig.h
#define DEFAULT_BUFFER_SIZE 256

STATIC mp_obj_t stream_readall(mp_obj_t self_in);

#if MICROPY_STREAMS_NON_BLOCK
// TODO: This is POSIX-specific (but then POSIX is the only real thing,
// and anything else just emulates it, right?)
#define is_nonblocking_error(errno) ((errno) == EAGAIN || (errno) == EWOULDBLOCK)
#else
#define is_nonblocking_error(errno) (0)
#endif

#define STREAM_CONTENT_TYPE(stream) (((stream)->is_text) ? &mp_type_str : &mp_type_bytes)

STATIC mp_obj_t stream_read(uint n_args, const mp_obj_t *args) {
    struct _mp_obj_base_t *o = (struct _mp_obj_base_t *)args[0];
    if (o->type->stream_p == NULL || o->type->stream_p->read == NULL) {
        // CPython: io.UnsupportedOperation, OSError subclass
        nlr_raise(mp_obj_new_exception_msg(&mp_type_OSError, "Operation not supported"));
    }

    // What to do if sz < -1?  Python docs don't specify this case.
    // CPython does a readall, but here we silently let negatives through,
    // and they will cause a MemoryError.
    mp_int_t sz;
    if (n_args == 1 || ((sz = mp_obj_get_int(args[1])) == -1)) {
        return stream_readall(args[0]);
    }

    #if MICROPY_PY_BUILTINS_STR_UNICODE
    if (o->type->stream_p->is_text) {
        // We need to read sz number of unicode characters.  Because we don't have any
        // buffering, and because the stream API can only read bytes, we must read here
        // in units of bytes and must never over read.  If we want sz chars, then reading
        // sz bytes will never over-read, so we follow this approach, in a loop to keep
        // reading until we have exactly enough chars.  This will be 1 read for text
        // with ASCII-only chars, and about 2 reads for text with a couple of non-ASCII
        // chars.  For text with lots of non-ASCII chars, it'll be pretty inefficient
        // in time and memory.

        vstr_t vstr;
        vstr_init(&vstr, sz);
        mp_uint_t more_bytes = sz;
        mp_uint_t last_buf_offset = 0;
        while (more_bytes > 0) {
            char *p = vstr_add_len(&vstr, more_bytes);
            if (p == NULL) {
                nlr_raise(mp_obj_new_exception_msg_varg(&mp_type_MemoryError, "out of memory"));
            }
            int error;
            mp_uint_t out_sz = o->type->stream_p->read(o, p, more_bytes, &error);
            if (out_sz == MP_STREAM_ERROR) {
                vstr_cut_tail_bytes(&vstr, more_bytes);
                if (is_nonblocking_error(error)) {
                    // With non-blocking streams, we read as much as we can.
                    // If we read nothing, return None, just like read().
                    // Otherwise, return data read so far.
                    // TODO what if we have read only half a non-ASCII char?
                    if (vstr.len == 0) {
                        vstr_clear(&vstr);
                        return mp_const_none;
                    }
                    break;
                }
                nlr_raise(mp_obj_new_exception_arg1(&mp_type_OSError, MP_OBJ_NEW_SMALL_INT(error)));
            }

            if (out_sz < more_bytes) {
                // Finish reading.
                // TODO what if we have read only half a non-ASCII char?
                vstr_cut_tail_bytes(&vstr, more_bytes - out_sz);
                if (out_sz == 0) {
                    break; 
                }
            }

            // count chars from bytes just read
            for (mp_uint_t off = last_buf_offset;;) {
                byte b = vstr.buf[off];
                int n;
                if (!UTF8_IS_NONASCII(b)) {
                    // 1-byte ASCII char
                    n = 1;
                } else if ((b & 0xe0) == 0xc0) {
                    // 2-byte char
                    n = 2;
                } else if ((b & 0xf0) == 0xe0) {
                    // 3-byte char
                    n = 3;
                } else if ((b & 0xf8) == 0xf0) {
                    // 4-byte char
                    n = 4;
                } else {
                    // TODO
                    n = 5;
                }
                if (off + n <= vstr.len) {
                    // got a whole char in n bytes
                    off += n;
                    sz -= 1;
                    last_buf_offset = off;
                    if (off >= vstr.len) {
                        more_bytes = sz;
                        break;
                    }
                } else {
                    // didn't get a whole char, so work out how many extra bytes are needed for
                    // this partial char, plus bytes for additional chars that we want
                    more_bytes = (off + n - vstr.len) + (sz - 1);
                    break;
                }
            }
        }
