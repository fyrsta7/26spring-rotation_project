ZEND_API void zend_hash_copy(HashTable *target, HashTable *source, copy_ctor_func_t pCopyConstructor, void *tmp, uint size)
{
	Bucket *p;
	void *new_entry;

	IS_CONSISTENT(source);
	IS_CONSISTENT(target);

	p = source->pListHead;
	while (p) {
		if (p->nKeyLength) {
			zend_hash_update(target, p->arKey, p->nKeyLength, p->pData, size, &new_entry);
		} else {
			zend_hash_index_update(target, p->h, p->pData, size, &new_entry);
		}
        if (pCopyConstructor) {
            pCopyConstructor(new_entry);
        }
		p = p->pListNext;
	}
	target->pInternalPointer = target->pListHead;
}


