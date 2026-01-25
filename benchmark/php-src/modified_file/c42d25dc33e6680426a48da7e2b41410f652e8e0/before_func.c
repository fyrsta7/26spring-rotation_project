static void sdl_serialize_type(sdlTypePtr type, HashTable *tmp_encoders, HashTable *tmp_types, smart_str *out)
{
	int i;
	HashTable *tmp_elements = NULL;

	WSDL_CACHE_PUT_1(type->kind, out);
	sdl_serialize_string(type->name, out);
	sdl_serialize_string(type->namens, out);
	sdl_serialize_string(type->def, out);
	sdl_serialize_string(type->fixed, out);
	sdl_serialize_string(type->ref, out);
	WSDL_CACHE_PUT_1(type->nillable, out);
	WSDL_CACHE_PUT_1(type->form, out);
	sdl_serialize_encoder_ref(type->encode, tmp_encoders, out);

	if (type->restrictions) {
		WSDL_CACHE_PUT_1(1, out);
		sdl_serialize_resriction_int(type->restrictions->minExclusive,out);
		sdl_serialize_resriction_int(type->restrictions->minInclusive,out);
		sdl_serialize_resriction_int(type->restrictions->maxExclusive,out);
		sdl_serialize_resriction_int(type->restrictions->maxInclusive,out);
		sdl_serialize_resriction_int(type->restrictions->totalDigits,out);
		sdl_serialize_resriction_int(type->restrictions->fractionDigits,out);
		sdl_serialize_resriction_int(type->restrictions->length,out);
		sdl_serialize_resriction_int(type->restrictions->minLength,out);
		sdl_serialize_resriction_int(type->restrictions->maxLength,out);
		sdl_serialize_resriction_char(type->restrictions->whiteSpace,out);
		sdl_serialize_resriction_char(type->restrictions->pattern,out);
		if (type->restrictions->enumeration) {
			i = zend_hash_num_elements(type->restrictions->enumeration);
		} else {
			i = 0;
		}
		WSDL_CACHE_PUT_INT(i, out);
		if (i > 0) {
			sdlRestrictionCharPtr *tmp;

			zend_hash_internal_pointer_reset(type->restrictions->enumeration);
			while (zend_hash_get_current_data(type->restrictions->enumeration, (void**)&tmp) == SUCCESS) {
				sdl_serialize_resriction_char(*tmp, out);
				sdl_serialize_key(type->restrictions->enumeration, out);
				zend_hash_move_forward(type->restrictions->enumeration);
			}
		}
	} else {
		WSDL_CACHE_PUT_1(0, out);
	}
	if (type->elements) {
		i = zend_hash_num_elements(type->elements);
	} else {
		i = 0;
	}
	WSDL_CACHE_PUT_INT(i, out);
	if (i > 0) {
		sdlTypePtr *tmp;

	  tmp_elements = emalloc(sizeof(HashTable));
	  zend_hash_init(tmp_elements, 0, NULL, NULL, 0);

		zend_hash_internal_pointer_reset(type->elements);
		while (zend_hash_get_current_data(type->elements, (void**)&tmp) == SUCCESS) {
			sdl_serialize_key(type->elements, out);
			sdl_serialize_type(*tmp, tmp_encoders, tmp_types, out);
			zend_hash_add(tmp_elements, (char*)tmp, sizeof(*tmp), &i, sizeof(int), NULL);
			i--;
			zend_hash_move_forward(type->elements);
		}
	}

	if (type->attributes) {
		i = zend_hash_num_elements(type->attributes);
	} else {
		i = 0;
	}
	WSDL_CACHE_PUT_INT(i, out);
	if (i > 0) {
		sdlAttributePtr *tmp;
		zend_hash_internal_pointer_reset(type->attributes);
		while (zend_hash_get_current_data(type->attributes, (void**)&tmp) == SUCCESS) {
			sdl_serialize_key(type->attributes, out);
			sdl_serialize_attribute(*tmp, tmp_encoders, out);
			zend_hash_move_forward(type->attributes);
		}
	}
	if (type->model) {
		WSDL_CACHE_PUT_1(1, out);
		sdl_serialize_model(type->model, tmp_types, tmp_elements, out);
	} else {
		WSDL_CACHE_PUT_1(0, out);
	}
	if (tmp_elements != NULL) {
		zend_hash_destroy(tmp_elements);
		efree(tmp_elements);
	}
}