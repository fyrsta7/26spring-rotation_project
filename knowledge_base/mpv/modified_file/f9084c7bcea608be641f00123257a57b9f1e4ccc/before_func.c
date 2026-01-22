    resize_append(talloc_ctx, s, append.len + 1);
    memcpy(s->start + s->len, append.start, append.len);
    s->len += append.len;
    s->start[s->len] = '\0';
}

void bstr_xappend_asprintf(void *talloc_ctx, bstr *s, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    bstr_xappend_vasprintf(talloc_ctx, s, fmt, ap);
    va_end(ap);
}

// Exactly as bstr_xappend(), but with a formatted string.
void bstr_xappend_vasprintf(void *talloc_ctx, bstr *s, const char *fmt,
                            va_list ap)
