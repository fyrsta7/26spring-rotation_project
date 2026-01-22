                           zval* arena) {
  if (!msg) {
    ZVAL_NULL(val);
    return;
  }

  if (!ObjCache_Get(msg, val)) {
    Message* intern = emalloc(sizeof(Message));
    Message_SuppressDefaultProperties(desc->class_entry);
    zend_object_std_init(&intern->std, desc->class_entry);
    intern->std.handlers = &message_object_handlers;
    ZVAL_COPY(&intern->arena, arena);
    intern->desc = desc;
    intern->msg = msg;
    ZVAL_OBJ(val, &intern->std);
    ObjCache_Add(intern->msg, &intern->std);
  }
}

bool Message_GetUpbMessage(zval* val, const Descriptor* desc, upb_Arena* arena,
                           upb_Message** msg) {
  PBPHP_ASSERT(desc);

  if (Z_ISREF_P(val)) {
    ZVAL_DEREF(val);
  }

  if (Z_TYPE_P(val) == IS_OBJECT &&
      instanceof_function(Z_OBJCE_P(val), desc->class_entry)) {
    Message* intern = (Message*)Z_OBJ_P(val);
    upb_Arena_Fuse(arena, Arena_Get(&intern->arena));
    *msg = intern->msg;
    return true;
  } else {
    zend_throw_exception_ex(zend_ce_type_error, 0,
                            "Given value is not an instance of %s.",
