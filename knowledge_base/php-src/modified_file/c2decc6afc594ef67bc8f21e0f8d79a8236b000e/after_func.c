			} else if (Z_TYPE(ht->arData[idx].val) != IS_UNDEF) {
				ht->nInternalPointer = idx;
				break;
			}
		}
	}
	if (p->key) {
		zend_string_release(p->key);
	}
	if (ht->pDestructor) {
		zval tmp;
		ZVAL_COPY_VALUE(&tmp, &p->val);
		ZVAL_UNDEF(&p->val);
		ht->pDestructor(&tmp);
	} else {
		ZVAL_UNDEF(&p->val);
	}
}

