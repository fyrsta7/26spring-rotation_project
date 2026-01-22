		} else if (is_missing_required_utf_bom(enc, data, len)) {
			const char *error_msg = _(
				"BOM is required in '%s' if encoded as %s");
			const char *advise_msg = _(
				"The file '%s' is missing a byte order "
				"mark (BOM). Please use UTF-%sBE or UTF-%sLE "
				"(depending on the byte order) as "
				"working-tree-encoding.");
			advise(advise_msg, path, stripped, stripped);
			if (die_on_error)
				die(error_msg, path, enc);
			else {
				return error(error_msg, path, enc);
			}
		}

	}
	return 0;
}

static void trace_encoding(const char *context, const char *path,
			   const char *encoding, const char *buf, size_t len)
