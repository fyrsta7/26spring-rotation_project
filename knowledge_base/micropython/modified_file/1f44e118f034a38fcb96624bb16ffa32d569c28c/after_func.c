
#include "mpconfig.h"
#include "nlr.h"
#include "misc.h"
#include "qstr.h"
#include "obj.h"
#include "runtime0.h"
#include "runtime.h"

// Helpers for sequence types

#define SWAP(type, var1, var2) { type t = var2; var2 = var1; var1 = t; }

// Implements backend of sequence * integer operation. Assumes elements are
// memory-adjacent in sequence.
void mp_seq_multiply(const void *items, uint item_sz, uint len, uint times, void *dest) {
    for (int i = 0; i < times; i++) {
        uint copy_sz = item_sz * len;
        memcpy(dest, items, copy_sz);
        dest = (char*)dest + copy_sz;
    }
}

#if MICROPY_PY_BUILTINS_SLICE

bool mp_seq_get_fast_slice_indexes(machine_uint_t len, mp_obj_t slice, mp_bound_slice_t *indexes) {
    mp_obj_t ostart, ostop, ostep;
    machine_int_t start, stop;
    mp_obj_slice_get(slice, &ostart, &ostop, &ostep);

    if (ostart == mp_const_none) {
        start = 0;
    } else {
        start = MP_OBJ_SMALL_INT_VALUE(ostart);
    }
    if (ostop == mp_const_none) {
        stop = len;
    } else {
        stop = MP_OBJ_SMALL_INT_VALUE(ostop);
    }

    // Unlike subscription, out-of-bounds slice indexes are never error
    if (start < 0) {
        start = len + start;
        if (start < 0) {
            start = 0;
