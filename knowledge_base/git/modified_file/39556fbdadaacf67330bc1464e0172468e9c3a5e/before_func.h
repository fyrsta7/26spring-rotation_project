 * This must be called twice on the delta data buffer, first to get the
 * expected reference buffer size, and again to get the result buffer size.
 */
static inline unsigned long get_delta_hdr_size(const unsigned char **datap)
{
	const unsigned char *data = *datap;
	unsigned char cmd = *data++;
	unsigned long size = cmd & ~0x80;
	int i = 7;
	while (cmd & 0x80) {
		cmd = *data++;
		size |= (cmd & ~0x80) << i;
		i += 7;
	}
