bool PyUpb_PyToUpb(PyObject* obj, const upb_FieldDef* f, upb_MessageValue* val,
                   upb_Arena* arena) {
  switch (upb_FieldDef_CType(f)) {
    case kUpb_CType_Enum:
      return PyUpb_PyToUpbEnum(obj, upb_FieldDef_EnumSubDef(f), val);
    case kUpb_CType_Int32:
      return PyUpb_GetInt32(obj, &val->int32_val);
    case kUpb_CType_Int64:
      return PyUpb_GetInt64(obj, &val->int64_val);
    case kUpb_CType_UInt32:
      return PyUpb_GetUint32(obj, &val->uint32_val);
    case kUpb_CType_UInt64:
      return PyUpb_GetUint64(obj, &val->uint64_val);
    case kUpb_CType_Float:
      if (!PyFloat_Check(obj) && PyUpb_IsNumpyNdarray(obj, f)) return false;
      val->float_val = PyFloat_AsDouble(obj);
      return !PyErr_Occurred();
    case kUpb_CType_Double:
      if (!PyFloat_Check(obj) && PyUpb_IsNumpyNdarray(obj, f)) return false;
      val->double_val = PyFloat_AsDouble(obj);
      return !PyErr_Occurred();
    case kUpb_CType_Bool:
      if (!PyBool_Check(obj) && PyUpb_IsNumpyNdarray(obj, f)) return false;
      val->bool_val = PyLong_AsLong(obj);
      return !PyErr_Occurred();
    case kUpb_CType_Bytes: {
      char* ptr;
      Py_ssize_t size;
      if (PyBytes_AsStringAndSize(obj, &ptr, &size) < 0) return false;
      *val = PyUpb_MaybeCopyString(ptr, size, arena);
      return true;
    }
    case kUpb_CType_String: {
      Py_ssize_t size;
      if (PyBytes_Check(obj)) {
        // Use the object's bytes if they are valid UTF-8.
        char* ptr;
        if (PyBytes_AsStringAndSize(obj, &ptr, &size) < 0) return false;
        if (!utf8_range_IsValid(ptr, size)) {
          // Invalid UTF-8.  Try to convert the message to a Python Unicode
          // object, even though we know this will fail, just to get the
          // idiomatic Python error message.
          obj = PyUnicode_FromEncodedObject(obj, "utf-8", NULL);
          assert(!obj);
          return false;
        }
        *val = PyUpb_MaybeCopyString(ptr, size, arena);
        return true;
      } else {
        const char* ptr;
        ptr = PyUnicode_AsUTF8AndSize(obj, &size);
        if (PyErr_Occurred()) return false;
        *val = PyUpb_MaybeCopyString(ptr, size, arena);
        return true;
      }
    }
    case kUpb_CType_Message:
      PyErr_Format(PyExc_ValueError, "Message objects may not be assigned");
      return false;
    default:
      PyErr_Format(PyExc_SystemError,
                   "Getting a value from a field of unknown type %d",
                   upb_FieldDef_CType(f));
      return false;
  }
}