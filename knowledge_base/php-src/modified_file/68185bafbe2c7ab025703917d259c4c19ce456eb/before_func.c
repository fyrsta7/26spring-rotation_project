		} else {
			t = _zend_hash_append_ptr(target, p->key, Z_PTR(p->val));
			if (pCopyConstructor) {
				pCopyConstructor(&Z_PTR_P(t));
			}
		}
	}
	target->nInternalPointer = target->nNumOfElements ? 0 : HT_INVALID_IDX;
	return;

failure:
	ce1 = Z_PTR(p->val);
	CG(in_compilation) = 1;
	zend_set_compiled_filename(ce1->info.user.filename);
	CG(zend_lineno) = ce1->info.user.line_start;
	zend_error(E_ERROR, "Cannot declare %s %s, because the name is already in use", zend_get_object_type(ce1), ZSTR_VAL(ce1->name));
}

#ifdef __SSE2__
#include <mmintrin.h>
#include <emmintrin.h>
