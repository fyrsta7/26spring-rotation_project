		vlen = vsep - ++ksep;
	} else {
		ksep = "";
		/* "foo&" */
		klen = vsep - var->ptr;
		vlen = 0;
	}

	php_url_decode(var->ptr, klen);

	val = estrndup(ksep, vlen);
	if (vlen) {
		vlen = php_url_decode(val, vlen);
	}

	if (sapi_module.input_filter(PARSE_POST, var->ptr, &val, vlen, &new_vlen)) {
		php_register_variable_safe(var->ptr, val, new_vlen, arr);
	}
	efree(val);

	var->ptr = vsep + (vsep != var->end);
