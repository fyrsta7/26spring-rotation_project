
			url_len = strlen(url);
			for (i = url_len - 1; url[i] == '/' && 0 <= i; i--)
				;
			url_len = i + 1;
			if (4 < i && !strncmp(".git", url + i - 3, 4))
				url_len = i - 3;

			strbuf_reset(&note);
			if (*what) {
				if (*kind)
					strbuf_addf(&note, "%s ", kind);
				strbuf_addf(&note, "'%s' of ", what);
			}

			append_fetch_head(&fetch_head, &rm->old_oid,
					  rm->fetch_head_status,
					  note.buf, url, url_len);

			strbuf_reset(&note);
			if (ref) {
				rc |= update_local_ref(ref, transaction, what,
						       rm, &note, summary_width);
				free(ref);
			} else if (write_fetch_head || dry_run) {
				/*
				 * Display fetches written to FETCH_HEAD (or
				 * would be written to FETCH_HEAD, if --dry-run
