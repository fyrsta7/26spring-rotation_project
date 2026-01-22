}

int BIO_dump(BIO *bp, const void *s, int len)
{
    return BIO_dump_cb(write_bio, bp, s, len);
}

int BIO_dump_indent(BIO *bp, const void *s, int len, int indent)
{
    return BIO_dump_indent_cb(write_bio, bp, s, len, indent);
}

int BIO_hex_string(BIO *out, int indent, int width, const void *data,
                   int datalen)
{
    const unsigned char *d = data;
    int i, j = 0;

    if (datalen < 1)
        return 1;

    for (i = 0; i < datalen - 1; i++) {
        if (i && !j)
            BIO_printf(out, "%*s", indent, "");

        BIO_printf(out, "%02X:", d[i]);
