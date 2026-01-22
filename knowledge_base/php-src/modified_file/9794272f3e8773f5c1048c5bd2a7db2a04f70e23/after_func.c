{
	uint32_t nIndex;
	uint32_t idx;
	Bucket *p, *arData;

	arData = ht->arData;
	nIndex = h | ht->nTableMask;
	idx = HT_HASH_EX(arData, nIndex);
	while (idx != HT_INVALID_IDX) {
		ZEND_ASSERT(idx < HT_IDX_TO_HASH(ht->nTableSize));
		p = HT_HASH_TO_BUCKET_EX(arData, idx);
		if (p->h == h && !p->key) {
			return p;
		}
		idx = Z_NEXT(p->val);
	}
	return NULL;
}

static zend_always_inline zval *_zend_hash_add_or_update_i(HashTable *ht, zend_string *key, zval *pData, uint32_t flag ZEND_FILE_LINE_DC)
{
	zend_ulong h;
	uint32_t nIndex;
	uint32_t idx;
	Bucket *p, *arData;

	IS_CONSISTENT(ht);
	HT_ASSERT_RC1(ht);

	if (UNEXPECTED(!(HT_FLAGS(ht) & HASH_FLAG_INITIALIZED))) {
		CHECK_INIT(ht, 0);
		if (!ZSTR_IS_INTERNED(key)) {
			zend_string_addref(key);
			HT_FLAGS(ht) &= ~HASH_FLAG_STATIC_KEYS;
			zend_string_hash_val(key);
		}
		goto add_to_hash;
	} else if (HT_FLAGS(ht) & HASH_FLAG_PACKED) {
		zend_hash_packed_to_hash(ht);
		if (!ZSTR_IS_INTERNED(key)) {
			zend_string_addref(key);
			HT_FLAGS(ht) &= ~HASH_FLAG_STATIC_KEYS;
			zend_string_hash_val(key);
		}
	} else if ((flag & HASH_ADD_NEW) == 0) {
		p = zend_hash_find_bucket(ht, key, 0);

		if (p) {
			zval *data;

			if (flag & HASH_ADD) {
				if (!(flag & HASH_UPDATE_INDIRECT)) {
					return NULL;
				}
				ZEND_ASSERT(&p->val != pData);
				data = &p->val;
				if (Z_TYPE_P(data) == IS_INDIRECT) {
					data = Z_INDIRECT_P(data);
					if (Z_TYPE_P(data) != IS_UNDEF) {
						return NULL;
					}
				} else {
					return NULL;
				}
			} else {
				ZEND_ASSERT(&p->val != pData);
				data = &p->val;
				if ((flag & HASH_UPDATE_INDIRECT) && Z_TYPE_P(data) == IS_INDIRECT) {
					data = Z_INDIRECT_P(data);
				}
			}
			if (ht->pDestructor) {
				ht->pDestructor(data);
			}
			ZVAL_COPY_VALUE(data, pData);
			return data;
		}
		if (!ZSTR_IS_INTERNED(key)) {
			zend_string_addref(key);
			HT_FLAGS(ht) &= ~HASH_FLAG_STATIC_KEYS;
		}
	} else if (!ZSTR_IS_INTERNED(key)) {
		zend_string_addref(key);
		HT_FLAGS(ht) &= ~HASH_FLAG_STATIC_KEYS;
		zend_string_hash_val(key);
	}

	ZEND_HASH_IF_FULL_DO_RESIZE(ht);		/* If the Hash table is full, resize it */
