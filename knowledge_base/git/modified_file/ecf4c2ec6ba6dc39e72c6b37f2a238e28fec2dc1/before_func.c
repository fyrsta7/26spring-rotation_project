	if (offset < 0)
		return offset;

	patch->ws_rule = whitespace_rule(patch->new_name
					 ? patch->new_name
					 : patch->old_name);

	patchsize = parse_single_patch(buffer + offset + hdrsize,
				       size - offset - hdrsize, patch);

	if (!patchsize) {
		static const char *binhdr[] = {
			"Binary files ",
			"Files ",
			NULL,
		};
		static const char git_binary[] = "GIT binary patch\n";
		int i;
		int hd = hdrsize + offset;
		unsigned long llen = linelen(buffer + hd, size - hd);

		if (llen == sizeof(git_binary) - 1 &&
		    !memcmp(git_binary, buffer + hd, llen)) {
			int used;
			linenr++;
			used = parse_binary(buffer + hd + llen,
					    size - hd - llen, patch);
			if (used)
				patchsize = used + llen;
			else
				patchsize = 0;
		}
		else if (!memcmp(" differ\n", buffer + hd + llen - 8, 8)) {
			for (i = 0; binhdr[i]; i++) {
				int len = strlen(binhdr[i]);
				if (len < size - hd &&
				    !memcmp(binhdr[i], buffer + hd, len)) {
					linenr++;
					patch->is_binary = 1;
					patchsize = llen;
					break;
				}
			}
		}

		/* Empty patch cannot be applied if it is a text patch
		 * without metadata change.  A binary patch appears
		 * empty to us here.
		 */
		if ((apply || check) &&
		    (!patch->is_binary && !metadata_changes(patch)))
			die("patch with only garbage at line %d", linenr);
	}

	return offset + hdrsize + patchsize;
}

