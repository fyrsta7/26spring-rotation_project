		*bufp = buf;
	}
	*sizep = newsize;
	memcpy(buf + size, one_line, len);
}

static void check_valid(unsigned char *sha1, const char *expect)
{
	void *buf;
	char type[20];
	unsigned long size;
