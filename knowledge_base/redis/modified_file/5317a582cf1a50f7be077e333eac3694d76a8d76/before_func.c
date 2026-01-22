 * We need two steps, in one we need to clear the bits, and in the other
 * we need to bitwise-OR the new bits.
 *
 * This time let's try with 'pos' = 1, so our first byte at 'b' is 0,
 *
 * "ls" is 6, and you may notice it is actually the position of the first
 * bit inside the byte. "rs" is 8-ls = 2
 *
 *   +--------+
 *   |00000011|  <- Our initial byte at "b"
 *   +--------+
 *
 * To create a AND-mask to clear the bits about this position, we just
 * initialize the mask with 2^6-1, right shift it of "ls" bits, and invert
 * it.
 *
 *   +--------+
 *   |11111100|  <- "mask" starts at 2^6-1
 *   |00000011|  <- "mask" after right shift of "ls" bits.
 *   |11111100|  <- "mask" after invert.
 *   +--------+
 *
 * Now we can bitwise-AND the byte at "b" with the mask, and bitwise-OR
 * it with "val" right-shifted of "ls" bits to set the new bits.
 *
 * Now let's focus on the next byte b+1:
 *
 *   +--------+
 *   |11112222| <- byte at b+1
 *   +--------+
 *
 * To build the AND mask we start again with the 2^6-1 value, left shift
 * it by "rs" bits, and invert it.
 *
 *   +--------+
 *   |11111100|  <- "mask" set at 2&6-1
 *   |11110000|  <- "mask" after the left shift of "rs" bits.
 *   |00001111|  <- "mask" after bitwise not.
 *   +--------+
 *
 * Now we can mask it with b+1 to clear the old bits, and bitwise-OR
 * with "val" left-shifted by "rs" bits to set the new value.
 */

/* Note: if we access the last counter, we will also access the b+1 byte
 * that is out of the array, but sds strings always have an implicit null
 * term, so the byte exists, and we can skip the conditional (or the need
 * to allocate 1 byte more explicitly). */

/* Store the value of the register at position 'regnum' into variable 'target'.
 * 'p' is an array of unsigned bytes. */
#define HLL_GET_REGISTER(target,p,regnum) do { \
    uint8_t *_p = (uint8_t*) p; \
    unsigned long _byte = regnum*REDIS_HLL_BITS/8; \
    unsigned long _leftshift = regnum*REDIS_HLL_BITS&7; \
    unsigned long _rightshift = 8 - _leftshift; \
    target = ((_p[_byte] << _leftshift) | \
             (_p[_byte+1] >> _rightshift)) & \
             REDIS_HLL_REGISTER_MAX; \
} while(0)

/* Set the value of the register at position 'regnum' to 'val'.
 * 'p' is an array of unsigned bytes. */
#define HLL_SET_REGISTER(p,regnum,val) do { \
    uint8_t *_p = (uint8_t*) p; \
    unsigned long _byte = regnum*REDIS_HLL_BITS/8; \
    unsigned long _leftshift = regnum*REDIS_HLL_BITS&7; \
    unsigned long _rightshift = 8 - _leftshift; \
    _p[_byte] &= ~(REDIS_HLL_REGISTER_MAX >> _leftshift); \
    _p[_byte] |= val >> _leftshift; \
    _p[_byte+1] &= ~(REDIS_HLL_REGISTER_MAX << _rightshift); \
    _p[_byte+1] |= val << _rightshift; \
