static inline reflection_object *reflection_object_from_obj(zend_object *obj) {
	return (reflection_object*)((char*)(obj) - XtOffsetOf(reflection_object, zo));
}

#define Z_REFLECTION_P(zv)  reflection_object_from_obj(Z_OBJ_P((zv)))
/* }}} */

static zend_object_handlers reflection_object_handlers;
