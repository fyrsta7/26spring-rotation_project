	uint32_t idx = 0;
	Bucket *p = source->arData;
	Bucket *q = target->arData;
	Bucket *end = p + source->nNumUsed;

	do {
		if (!zend_array_dup_element(source, target, idx, p, q, 0, static_keys, with_holes)) {
			uint32_t target_idx = idx;

			idx++; p++;
			while (p != end) {
				if (zend_array_dup_element(source, target, target_idx, p, q, 0, static_keys, with_holes)) {
					if (source->nInternalPointer == idx) {
						target->nInternalPointer = target_idx;
					}
					target_idx++; q++;
				}
				idx++; p++;
			}
			return target_idx;
		}
		idx++; p++; q++;
	} while (p != end);
	return idx;
}

ZEND_API HashTable* ZEND_FASTCALL zend_array_dup(HashTable *source)
{
	uint32_t idx;
	HashTable *target;

	IS_CONSISTENT(source);

	ALLOC_HASHTABLE(target);
	GC_SET_REFCOUNT(target, 1);
	GC_TYPE_INFO(target) = IS_ARRAY | (GC_COLLECTABLE << GC_FLAGS_SHIFT);

	target->nTableSize = source->nTableSize;
	target->pDestructor = ZVAL_PTR_DTOR;

	if (source->nNumUsed == 0) {
		HT_FLAGS(target) = (HT_FLAGS(source) & ~(HASH_FLAG_INITIALIZED|HASH_FLAG_PACKED)) | HASH_FLAG_STATIC_KEYS;
		target->nTableMask = HT_MIN_MASK;
		target->nNumUsed = 0;
		target->nNumOfElements = 0;
		target->nNextFreeElement = 0;
		target->nInternalPointer = HT_INVALID_IDX;
		HT_SET_DATA_ADDR(target, &uninitialized_bucket);
	} else if (GC_FLAGS(source) & IS_ARRAY_IMMUTABLE) {
		HT_FLAGS(target) = HT_FLAGS(source);
		target->nTableMask = source->nTableMask;
		target->nNumUsed = source->nNumUsed;
		target->nNumOfElements = source->nNumOfElements;
		target->nNextFreeElement = source->nNextFreeElement;
		HT_SET_DATA_ADDR(target, emalloc(HT_SIZE(target)));
		target->nInternalPointer = source->nInternalPointer;
		memcpy(HT_GET_DATA_ADDR(target), HT_GET_DATA_ADDR(source), HT_USED_SIZE(source));
		if (target->nNumOfElements > 0 &&
		    target->nInternalPointer == HT_INVALID_IDX) {
			idx = 0;
			while (Z_TYPE(target->arData[idx].val) == IS_UNDEF) {
				idx++;
			}
			target->nInternalPointer = idx;
		}
	} else if (HT_FLAGS(source) & HASH_FLAG_PACKED) {
		HT_FLAGS(target) = HT_FLAGS(source);
		target->nTableMask = source->nTableMask;
		target->nNumUsed = source->nNumUsed;
		target->nNumOfElements = source->nNumOfElements;
		target->nNextFreeElement = source->nNextFreeElement;
		HT_SET_DATA_ADDR(target, emalloc(HT_SIZE(target)));
		target->nInternalPointer = source->nInternalPointer;
		HT_HASH_RESET_PACKED(target);

		if (HT_IS_WITHOUT_HOLES(target)) {
			zend_array_dup_packed_elements(source, target, 0);
		} else {
			zend_array_dup_packed_elements(source, target, 1);
		}
		if (target->nNumOfElements > 0 &&
		    target->nInternalPointer == HT_INVALID_IDX) {
			idx = 0;
			while (Z_TYPE(target->arData[idx].val) == IS_UNDEF) {
				idx++;
			}
			target->nInternalPointer = idx;
		}
	} else {
		HT_FLAGS(target) = HT_FLAGS(source);
		target->nTableMask = source->nTableMask;
		target->nNextFreeElement = source->nNextFreeElement;
