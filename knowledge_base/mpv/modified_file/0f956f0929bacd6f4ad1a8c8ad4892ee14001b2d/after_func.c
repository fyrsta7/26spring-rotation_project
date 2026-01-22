static struct ra_mapped_buffer *gl_create_mapped_buffer(struct ra *ra,
                                                        size_t size)
{
    struct ra_gl *p = ra->priv;
    GL *gl = p->gl;

    if (gl->version < 440)
        return NULL;

    struct ra_mapped_buffer *buf = talloc_zero(NULL, struct ra_mapped_buffer);
    buf->size = size;

    struct ra_mapped_buffer_gl *buf_gl = buf->priv =
        talloc_zero(NULL, struct ra_mapped_buffer_gl);

    unsigned flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT |
                     GL_MAP_COHERENT_BIT;

    gl->GenBuffers(1, &buf_gl->pbo);
    gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, buf_gl->pbo);
    gl->BufferStorage(GL_PIXEL_UNPACK_BUFFER, size, NULL, flags | GL_CLIENT_STORAGE_BIT);
    buf->data = gl->MapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, buf->size, flags);
    gl->BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    if (!buf->data) {
        gl_check_error(gl, ra->log, "mapping buffer");
        gl_destroy_mapped_buffer(ra, buf);
        return NULL;
    }

    return buf;
}
