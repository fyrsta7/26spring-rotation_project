 * This must be called twice on the delta data buffer, first to get the
 * expected reference buffer size, and again to get the result buffer size.
 */
static inline unsigned long get_delta_hdr_size(const unsigned char **datap)
{
	const unsigned char *data = *datap;
	unsigned char cmd;
	unsigned long size = 0;
	int i = 0;
	do {
		cmd = *data++;
		size |= (cmd & ~0x80) << i;
		i += 7;
	} while (cmd & 0x80);
