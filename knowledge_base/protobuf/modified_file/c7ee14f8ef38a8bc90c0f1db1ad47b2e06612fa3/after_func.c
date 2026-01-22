  } else if(ft == GOOGLE_PROTOBUF_FIELDDESCRIPTORPROTO_TYPE_GROUP) {
    /* No length specified, an "end group" tag will mark the end. */
    UPB_CHECK(push_stack_frame(s, UINT32_MAX, user_field_desc));
  } else {
    UPB_CHECK(s->value_cb(s, buf, end, user_field_desc));
  }
  return UPB_STATUS_OK;
}

upb_status_t upb_parse(struct upb_parse_state *restrict s, void *buf, size_t len,
                       size_t *restrict read)
{
  void *end = (char*)buf + len;
  *read = 0;
  while(buf < end) {
    struct upb_tag tag;
    void *bufstart = buf;
    UPB_CHECK(parse_tag(&buf, end, &tag));
    if(tag.wire_type == UPB_WIRE_TYPE_END_GROUP) {
      if(s->top->end_offset != UINT32_MAX)
        return UPB_ERROR_SPURIOUS_END_GROUP;
      pop_stack_frame(s);
    } else if(tag.wire_type == UPB_WIRE_TYPE_DELIMITED) {
      parse_delimited(s, &tag, &buf, end, s->offset + (char*)buf - (char*)bufstart);
    } else {
      parse_nondelimited(s, &tag, &buf, end);
    }
    size_t bytes_read = ((char*)buf - (char*)bufstart);
