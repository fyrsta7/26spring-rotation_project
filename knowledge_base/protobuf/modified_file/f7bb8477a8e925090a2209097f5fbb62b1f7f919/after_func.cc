void Generator::GenerateClassDeserializeBinaryField(
    const GeneratorOptions& options,
    io::Printer* printer,
    const FieldDescriptor* field) const {

  printer->Print("    case $num$:\n",
                 "num", SimpleItoa(field->number()));

  if (IsMap(options, field)) {
    const FieldDescriptor* key_field = MapFieldKey(field);
    const FieldDescriptor* value_field = MapFieldValue(field);
    printer->Print(
        "      var value = msg.get$name$();\n"
        "      reader.readMessage(value, function(message, reader) {\n",
        "name", JSGetterName(options, field));

    printer->Print("        jspb.Map.deserializeBinary(message, reader, "
                   "$keyReaderFn$, $valueReaderFn$",
          "keyReaderFn", JSBinaryReaderMethodName(options, key_field),
          "valueReaderFn", JSBinaryReaderMethodName(options, value_field));

    if (value_field->type() == FieldDescriptor::TYPE_MESSAGE) {
      printer->Print(", $messageType$.deserializeBinaryFromReader",
          "messageType", GetPath(options, value_field->message_type()));
    }

    printer->Print(");\n");
    printer->Print("         });\n");
  } else {
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      printer->Print(
          "      var value = new $fieldclass$;\n"
          "      reader.read$msgOrGroup$($grpfield$value,"
          "$fieldclass$.deserializeBinaryFromReader);\n",
        "fieldclass", SubmessageTypeRef(options, field),
          "msgOrGroup", (field->type() == FieldDescriptor::TYPE_GROUP) ?
                        "Group" : "Message",
          "grpfield", (field->type() == FieldDescriptor::TYPE_GROUP) ?
                      (SimpleItoa(field->number()) + ", ") : "");
    } else {
      printer->Print(
          "      var value = /** @type {$fieldtype$} */ "
          "(reader.read$reader$());\n",
          "fieldtype", JSFieldTypeAnnotation(options, field, false, true,
                                             /* singular_if_not_packed */ true,
                                             BYTES_U8),
          "reader",
          JSBinaryReadWriteMethodName(field, /* is_writer = */ false));
    }

    if (field->is_repeated() && !field->is_packed()) {
      printer->Print(
          "      msg.add$name$(value);\n", "name",
          JSGetterName(options, field, BYTES_DEFAULT, /* drop_list = */ true));
    } else {
      // Singular fields, and packed repeated fields, receive a |value| either
      // as the field's value or as the array of all the field's values; set
      // this as the field's value directly.
      printer->Print(
          "      msg.set$name$(value);\n",
          "name", JSGetterName(options, field));
    }
  }

  printer->Print("      break;\n");
}
