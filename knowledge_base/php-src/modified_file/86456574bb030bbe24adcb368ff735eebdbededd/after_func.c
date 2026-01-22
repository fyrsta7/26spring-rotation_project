		}
		iter++;
	}
}

ZEND_API void ZEND_FASTCALL zend_hash_iterators_advance(HashTable *ht, HashPosition step)
{
	HashTableIterator *iter = EG(ht_iterators);
	HashTableIterator *end  = iter + EG(ht_iterators_used);

	while (iter != end) {
		if (iter->ht == ht) {
			iter->pos += step;
		}
		iter++;
	}
}

/* Hash must be known and precomputed before */
static zend_always_inline Bucket *zend_hash_find_bucket(const HashTable *ht, const zend_string *key)
{
	uint32_t nIndex;
	uint32_t idx;
	Bucket *p, *arData;

	ZEND_ASSERT(ZSTR_H(key) != 0 && "Hash must be known");

	arData = ht->arData;
	nIndex = ZSTR_H(key) | ht->nTableMask;
	idx = HT_HASH_EX(arData, nIndex);

	if (UNEXPECTED(idx == HT_INVALID_IDX)) {
		return NULL;
	}
	p = HT_HASH_TO_BUCKET_EX(arData, idx);
	if (EXPECTED(p->key == key)) { /* check for the same interned string */
