  }
  return out;
}

static bool _upb_MessageDef_EncodeMap(upb_DescState* s, const upb_MessageDef* m,
                                      upb_Arena* a) {
  if (m->field_count != 2) return false;

  const upb_FieldDef* key_field = upb_MessageDef_Field(m, 0);
  const upb_FieldDef* val_field = upb_MessageDef_Field(m, 1);
  if (key_field == NULL || val_field == NULL) return false;

  UPB_ASSERT(_upb_FieldDef_LayoutIndex(key_field) == 0);
  UPB_ASSERT(_upb_FieldDef_LayoutIndex(val_field) == 1);

  const upb_FieldType key_type = upb_FieldDef_Type(key_field);
  const upb_FieldType val_type = upb_FieldDef_Type(val_field);

  const uint64_t val_mod = _upb_FieldDef_IsClosedEnum(val_field)
                               ? kUpb_FieldModifier_IsClosedEnum
                               : 0;

  s->ptr =
      upb_MtDataEncoder_EncodeMap(&s->e, s->ptr, key_type, val_type, val_mod);
  return true;
}

static bool _upb_MessageDef_EncodeMessage(upb_DescState* s,
                                          const upb_MessageDef* m,
                                          upb_Arena* a) {
  const upb_FieldDef** sorted = NULL;
  if (!m->is_sorted) {
    sorted = _upb_FieldDefs_Sorted(m->fields, m->field_count, a);
    if (!sorted) return false;
  }

  s->ptr = upb_MtDataEncoder_StartMessage(&s->e, s->ptr,
                                          _upb_MessageDef_Modifiers(m));
