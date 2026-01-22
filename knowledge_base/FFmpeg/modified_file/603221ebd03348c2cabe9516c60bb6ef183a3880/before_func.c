    return 0;
}

/**
 * Bitexact implementation of sqrt(val/2).
 */
static int16_t square_root(int val)
{
    return (ff_sqrt(val << 1) >> 1) & (~1);
}

/**
 * Calculate the number of left-shifts required for normalizing the input.
 *
 * @param num   input number
