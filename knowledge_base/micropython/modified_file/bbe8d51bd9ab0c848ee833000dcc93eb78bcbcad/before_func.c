    map->table = NULL;
}

STATIC void mp_map_rehash(mp_map_t *map) {
    mp_uint_t old_alloc = map->alloc;
    mp_map_elem_t *old_table = map->table;
    map->alloc = get_doubling_prime_greater_or_equal_to(map->alloc + 1);
    map->used = 0;
    map->all_keys_are_qstrs = 1;
    map->table = m_new0(mp_map_elem_t, map->alloc);
    for (mp_uint_t i = 0; i < old_alloc; i++) {
        if (old_table[i].key != MP_OBJ_NULL && old_table[i].key != MP_OBJ_SENTINEL) {
            mp_map_lookup(map, old_table[i].key, MP_MAP_LOOKUP_ADD_IF_NOT_FOUND)->value = old_table[i].value;
        }
    }
    m_del(mp_map_elem_t, old_table, old_alloc);
}

// MP_MAP_LOOKUP behaviour:
//  - returns NULL if not found, else the slot it was found in with key,value non-null
// MP_MAP_LOOKUP_ADD_IF_NOT_FOUND behaviour:
//  - returns slot, with key non-null and value=MP_OBJ_NULL if it was added
// MP_MAP_LOOKUP_REMOVE_IF_FOUND behaviour:
//  - returns NULL if not found, else the slot if was found in with key null and value non-null
mp_map_elem_t *mp_map_lookup(mp_map_t *map, mp_obj_t index, mp_map_lookup_kind_t lookup_kind) {

    // Work out if we can compare just pointers
    bool compare_only_ptrs = map->all_keys_are_qstrs;
    if (compare_only_ptrs) {
        if (MP_OBJ_IS_QSTR(index)) {
            // Index is a qstr, so can just do ptr comparison.
        } else if (MP_OBJ_IS_TYPE(index, &mp_type_str)) {
            // Index is a non-interned string.
            // We can either intern the string, or force a full equality comparison.
            // We chose the latter, since interning costs time and potentially RAM,
            // and it won't necessarily benefit subsequent calls because these calls
            // most likely won't pass the newly-interned string.
            compare_only_ptrs = false;
        } else if (lookup_kind != MP_MAP_LOOKUP_ADD_IF_NOT_FOUND) {
            // If we are not adding, then we can return straight away a failed
            // lookup because we know that the index will never be found.
            return NULL;
        }
    }

    // if the map is an ordered array then we must do a brute force linear search
    if (map->is_ordered) {
        if (map->is_fixed && lookup_kind != MP_MAP_LOOKUP) {
            // can't add/remove from a fixed array
            return NULL;
        }
        for (mp_map_elem_t *elem = &map->table[0], *top = &map->table[map->used]; elem < top; elem++) {
            if (elem->key == index || (!compare_only_ptrs && mp_obj_equal(elem->key, index))) {
                if (MP_UNLIKELY(lookup_kind == MP_MAP_LOOKUP_REMOVE_IF_FOUND)) {
                    elem->key = MP_OBJ_SENTINEL;
                    // keep elem->value so that caller can access it if needed
                }
                return elem;
            }
        }
        if (MP_LIKELY(lookup_kind != MP_MAP_LOOKUP_ADD_IF_NOT_FOUND)) {
            return NULL;
        }
        // TODO shrink array down over any previously-freed slots
        if (map->used == map->alloc) {
            // TODO: Alloc policy
            map->alloc += 4;
            map->table = m_renew(mp_map_elem_t, map->table, map->used, map->alloc);
            mp_seq_clear(map->table, map->used, map->alloc, sizeof(*map->table));
        }
        mp_map_elem_t *elem = map->table + map->used++;
        elem->key = index;
        if (!MP_OBJ_IS_QSTR(index)) {
            map->all_keys_are_qstrs = 0;
        }
        return elem;
    }

    // map is a hash table (not an ordered array), so do a hash lookup

    if (map->alloc == 0) {
        if (lookup_kind == MP_MAP_LOOKUP_ADD_IF_NOT_FOUND) {
            mp_map_rehash(map);
        } else {
            return NULL;
        }
    }

    mp_uint_t hash = MP_OBJ_SMALL_INT_VALUE(mp_unary_op(MP_UNARY_OP_HASH, index));
    mp_uint_t pos = hash % map->alloc;
    mp_uint_t start_pos = pos;
    mp_map_elem_t *avail_slot = NULL;
    for (;;) {
        mp_map_elem_t *slot = &map->table[pos];
        if (slot->key == MP_OBJ_NULL) {
            // found NULL slot, so index is not in table
            if (lookup_kind == MP_MAP_LOOKUP_ADD_IF_NOT_FOUND) {
                map->used += 1;
                if (avail_slot == NULL) {
                    avail_slot = slot;
                }
                avail_slot->key = index;
                avail_slot->value = MP_OBJ_NULL;
                if (!MP_OBJ_IS_QSTR(index)) {
                    map->all_keys_are_qstrs = 0;
                }
                return avail_slot;
            } else {
                return NULL;
            }
        } else if (slot->key == MP_OBJ_SENTINEL) {
            // found deleted slot, remember for later
            if (avail_slot == NULL) {
                avail_slot = slot;
            }
        } else if (slot->key == index || (!compare_only_ptrs && mp_obj_equal(slot->key, index))) {
            // found index
            // Note: CPython does not replace the index; try x={True:'true'};x[1]='one';x
            if (lookup_kind == MP_MAP_LOOKUP_REMOVE_IF_FOUND) {
                // delete element in this slot
                map->used--;
                if (map->table[(pos + 1) % map->alloc].key == MP_OBJ_NULL) {
                    // optimisation if next slot is empty
                    slot->key = MP_OBJ_NULL;
                } else {
                    slot->key = MP_OBJ_SENTINEL;
                }
                // keep slot->value so that caller can access it if needed
            }
            return slot;
        }

        // not yet found, keep searching in this table
        pos = (pos + 1) % map->alloc;

