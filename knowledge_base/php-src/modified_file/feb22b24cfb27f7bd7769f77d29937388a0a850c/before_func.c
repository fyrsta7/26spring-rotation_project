								iter_pos = zend_hash_iterators_lower_pos(ht, iter_pos + 1);
							}
							q++;
							j++;
						}
					}
				}
				ht->nNumUsed = j;
				break;
			}
			nIndex = p->h | ht->nTableMask;
			Z_NEXT(p->val) = HT_HASH(ht, nIndex);
			HT_HASH(ht, nIndex) = HT_IDX_TO_HASH(i);
			p++;
		} while (++i < ht->nNumUsed);
	}
	return SUCCESS;
}

static zend_always_inline void _zend_hash_del_el_ex(HashTable *ht, uint32_t idx, Bucket *p, Bucket *prev)
{
	if (!(HT_FLAGS(ht) & HASH_FLAG_PACKED)) {
		if (prev) {
			Z_NEXT(prev->val) = Z_NEXT(p->val);
		} else {
			HT_HASH(ht, p->h | ht->nTableMask) = Z_NEXT(p->val);
		}
	}
	if (HT_IDX_TO_HASH(ht->nNumUsed - 1) == idx) {
		do {
			ht->nNumUsed--;
		} while (ht->nNumUsed > 0 && (UNEXPECTED(Z_TYPE(ht->arData[ht->nNumUsed-1].val) == IS_UNDEF)));
	}
	ht->nNumOfElements--;
	if (HT_IDX_TO_HASH(ht->nInternalPointer) == idx || UNEXPECTED(HT_HAS_ITERATORS(ht))) {
		uint32_t new_idx;

		new_idx = idx = HT_HASH_TO_IDX(idx);
		while (1) {
			new_idx++;
			if (new_idx >= ht->nNumUsed) {
				new_idx = HT_INVALID_IDX;
				break;
			} else if (Z_TYPE(ht->arData[new_idx].val) != IS_UNDEF) {
				break;
