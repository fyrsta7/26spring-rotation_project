	for (dividend = 0; dividend < CRC8_TABLE_LEN; ++dividend) {
		/* Start with the dividend followed by zeros */
		remainder = dividend << (WIDTH - 8);

		/* Perform modulo-2 division, a bit at a time */
		for (bit = 8; bit > 0; --bit) {
			/* Try to divide the current data bit */
			if (remainder & TOPBIT)
				remainder = (remainder << 1) ^ POLYNOMIAL;
			else
				remainder = (remainder << 1);
		}

		/* Store the result into the table */
		crc8table[dividend] = remainder;
	}
}
