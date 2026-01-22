		_zend_hash_append_ptr_ex(target, p->key, Z_PTR(p->val), 1);
	}
	target->nInternalPointer = 0;
	return;

failure:
	function1 = Z_PTR(p->val);
	function2 = Z_PTR_P(t);
	CG(in_compilation) = 1;
	zend_set_compiled_filename(function1->op_array.filename);
	CG(zend_lineno) = function1->op_array.opcodes[0].lineno;
	if (function2->type == ZEND_USER_FUNCTION
		&& function2->op_array.last > 0) {
		zend_error(E_ERROR, "Cannot redeclare %s() (previously declared in %s:%d)",
				   ZSTR_VAL(function1->common.function_name),
				   ZSTR_VAL(function2->op_array.filename),
				   (int)function2->op_array.opcodes[0].lineno);
	} else {
		zend_error(E_ERROR, "Cannot redeclare %s()", ZSTR_VAL(function1->common.function_name));
	}
}

static void zend_accel_class_hash_copy(HashTable *target, HashTable *source)
{
	Bucket *p, *end;
	zval *t;

	zend_hash_extend(target, target->nNumUsed + source->nNumUsed, 0);
	p = source->arData;
	end = p + source->nNumUsed;
	for (; p != end; p++) {
		ZEND_ASSERT(Z_TYPE(p->val) != IS_UNDEF);
		ZEND_ASSERT(p->key);
		t = zend_hash_find_ex(target, p->key, 1);
		if (UNEXPECTED(t != NULL)) {
			if (EXPECTED(ZSTR_LEN(p->key) > 0) && EXPECTED(ZSTR_VAL(p->key)[0] == 0)) {
				/* Runtime definition key. There are two circumstances under which the key can
				 * already be defined:
				 *  1. The file has been re-included without being changed in the meantime. In
				 *     this case we can keep the old value, because we know that the definition
				 *     hasn't changed.
				 *  2. The file has been changed in the meantime, but the RTD key ends up colliding.
