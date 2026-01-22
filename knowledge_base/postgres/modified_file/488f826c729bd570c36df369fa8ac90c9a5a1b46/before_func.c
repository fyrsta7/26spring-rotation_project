				}
			}
		}
		else
		{
			RelFileLocator locator;

			locator = BufTagGetRelFileLocator(&bufHdr->tag);
			rlocator = bsearch((const void *) &(locator),
							   locators, n, sizeof(RelFileLocator),
							   rlocator_comparator);
		}

		/* buffer doesn't belong to any of the given relfilelocators; skip it */
		if (rlocator == NULL)
			continue;

		buf_state = LockBufHdr(bufHdr);
		if (BufTagMatchesRelFileLocator(&bufHdr->tag, rlocator))
			InvalidateBuffer(bufHdr);	/* releases spinlock */
		else
			UnlockBufHdr(bufHdr, buf_state);
	}
