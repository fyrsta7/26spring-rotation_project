		tracker->str = NULL;
	}
	if (!Z_ISUNDEF(tracker->val)) {
		/* Here, copy the original zval to a different pointer without incrementing the refcount in case something uses the original while it's being freed. */
		zval zval_copy;

		ZEND_ASSERT(!persistent);
		ZVAL_COPY_VALUE(&zval_copy, &tracker->val);
		ZVAL_UNDEF(&tracker->val);
		zval_ptr_dtor(&zval_copy);
	}
}
/* }}} */

/**
 * Free memory used to track the metadata and set all fields to be null/undef.
 */
void phar_metadata_tracker_copy(phar_metadata_tracker *dest, const phar_metadata_tracker *source, bool persistent) /* {{{ */
{
	ZEND_ASSERT(dest != source);
	phar_metadata_tracker_free(dest, persistent);

	if (!Z_ISUNDEF(source->val)) {
		ZEND_ASSERT(!persistent);
		ZVAL_COPY(&dest->val, &source->val);
	}
	if (source->str) {
		dest->str = zend_string_copy(source->str);
	}
}
/* }}} */

/**
 * Copy constructor for a non-persistent clone.
 */
void phar_metadata_tracker_clone(phar_metadata_tracker *tracker) /* {{{ */
{
	Z_TRY_ADDREF_P(&tracker->val);
	if (tracker->str) {
		/* Duplicate the string, as the original may have been persistent. */
		tracker->str = zend_string_dup(tracker->str, false);
	}
}
/* }}} */

/**
 * Parse out metadata from the manifest for a single file, saving it into a string.
 *
 * Meta-data is in this format:
 * [len32][data...]
 *
 * data is the serialized zval
 */
void phar_parse_metadata_lazy(const char *buffer, phar_metadata_tracker *tracker, uint32_t zip_metadata_len, bool persistent) /* {{{ */
{
	phar_metadata_tracker_free(tracker, persistent);
	if (zip_metadata_len) {
		/* lazy init metadata */
		tracker->str = zend_string_init(buffer, zip_metadata_len, persistent);
	}
}
/* }}}*/

/**
 * Size of fixed fields in the manifest.
 * See: https://www.php.net/manual/en/phar.fileformat.phar.php
 */
#define MANIFEST_FIXED_LEN	18

#define SAFE_PHAR_GET_32(buffer, endbuffer, var) \
	if (UNEXPECTED(buffer + 4 > endbuffer)) { \
		MAPPHAR_FAIL("internal corruption of phar \"%s\" (truncated manifest header)"); \
	} \
	PHAR_GET_32(buffer, var);

/**
 * Does not check for a previously opened phar in the cache.
 *
 * Parse a new one and add it to the cache, returning either SUCCESS or
 * FAILURE, and setting pphar to the pointer to the manifest entry
 *
 * This is used by phar_open_from_filename to process the manifest, but can be called
 * directly.
 */
static zend_result phar_parse_pharfile(php_stream *fp, char *fname, size_t fname_len, char *alias, size_t alias_len, zend_long halt_offset, phar_archive_data** pphar, uint32_t compression, char **error) /* {{{ */
{
	char b32[4], *buffer, *endbuffer, *savebuf;
	phar_archive_data *mydata = NULL;
	phar_entry_info entry;
	uint32_t manifest_len, manifest_count, manifest_flags, manifest_index, tmp_len, sig_flags;
	uint16_t manifest_ver;
	uint32_t len;
	zend_long offset;
	size_t sig_len;
	int register_alias = 0, temp_alias = 0;
	char *signature = NULL;
	zend_string *str;

	if (pphar) {
		*pphar = NULL;
	}

	if (error) {
		*error = NULL;
	}

	/* check for ?>\n and increment accordingly */
	if (-1 == php_stream_seek(fp, halt_offset, SEEK_SET)) {
		MAPPHAR_ALLOC_FAIL("cannot seek to __HALT_COMPILER(); location in phar \"%s\"")
	}

	buffer = b32;

	if (3 != php_stream_read(fp, buffer, 3)) {
		MAPPHAR_ALLOC_FAIL("internal corruption of phar \"%s\" (truncated manifest at stub end)")
	}

	if ((*buffer == ' ' || *buffer == '\n') && *(buffer + 1) == '?' && *(buffer + 2) == '>') {
		int nextchar;
		halt_offset += 3;
		if (EOF == (nextchar = php_stream_getc(fp))) {
			MAPPHAR_ALLOC_FAIL("internal corruption of phar \"%s\" (truncated manifest at stub end)")
		}

		if ((char) nextchar == '\r') {
			/* if we have an \r we require an \n as well */
			if (EOF == (nextchar = php_stream_getc(fp)) || (char)nextchar != '\n') {
				MAPPHAR_ALLOC_FAIL("internal corruption of phar \"%s\" (truncated manifest at stub end)")
			}
			++halt_offset;
		}

		if ((char) nextchar == '\n') {
			++halt_offset;
		}
	}

	/* make sure we are at the right location to read the manifest */
	if (-1 == php_stream_seek(fp, halt_offset, SEEK_SET)) {
		MAPPHAR_ALLOC_FAIL("cannot seek to __HALT_COMPILER(); location in phar \"%s\"")
	}

	/* read in manifest */
	buffer = b32;

	if (4 != php_stream_read(fp, buffer, 4)) {
		MAPPHAR_ALLOC_FAIL("internal corruption of phar \"%s\" (truncated manifest at manifest length)")
	}

	PHAR_GET_32(buffer, manifest_len);

	if (manifest_len > 1048576 * 100) {
		/* prevent serious memory issues by limiting manifest to at most 100 MB in length */
		MAPPHAR_ALLOC_FAIL("manifest cannot be larger than 100 MB in phar \"%s\"")
	}

	buffer = (char *)emalloc(manifest_len);
	savebuf = buffer;
	endbuffer = buffer + manifest_len;

	if (manifest_len < MANIFEST_FIXED_LEN || manifest_len != php_stream_read(fp, buffer, manifest_len)) {
		MAPPHAR_FAIL("internal corruption of phar \"%s\" (truncated manifest header)")
	}

	/* extract the number of entries */
	SAFE_PHAR_GET_32(buffer, endbuffer, manifest_count);

	if (manifest_count == 0) {
		MAPPHAR_FAIL("in phar \"%s\", manifest claims to have zero entries.  Phars must have at least 1 entry");
	}

	/* extract API version, lowest nibble currently unused */
	manifest_ver = (((unsigned char)buffer[0]) << 8)
				 + ((unsigned char)buffer[1]);
	buffer += 2;

	if ((manifest_ver & PHAR_API_VER_MASK) < PHAR_API_MIN_READ) {
		efree(savebuf);
		php_stream_close(fp);
		if (error) {
			spprintf(error, 0, "phar \"%s\" is API version %1.u.%1.u.%1.u, and cannot be processed", fname, manifest_ver >> 12, (manifest_ver >> 8) & 0xF, (manifest_ver >> 4) & 0x0F);
		}
		return FAILURE;
	}

	SAFE_PHAR_GET_32(buffer, endbuffer, manifest_flags);

	manifest_flags &= ~PHAR_HDR_COMPRESSION_MASK;
	manifest_flags &= ~PHAR_FILE_COMPRESSION_MASK;
	/* remember whether this entire phar was compressed with gz/bzip2 */
	manifest_flags |= compression;

	/* The lowest nibble contains the phar wide flags. The compression flags can */
	/* be ignored on reading because it is being generated anyways. */
	if (manifest_flags & PHAR_HDR_SIGNATURE) {
		char sig_buf[8], *sig_ptr = sig_buf;
		zend_off_t read_len;
		size_t end_of_phar;

		if (-1 == php_stream_seek(fp, -8, SEEK_END)
		|| (read_len = php_stream_tell(fp)) < 20
		|| 8 != php_stream_read(fp, sig_buf, 8)
		|| memcmp(sig_buf+4, "GBMB", 4)) {
			efree(savebuf);
			php_stream_close(fp);
			if (error) {
				spprintf(error, 0, "phar \"%s\" has a broken signature", fname);
			}
			return FAILURE;
		}

		PHAR_GET_32(sig_ptr, sig_flags);

		switch(sig_flags) {
			case PHAR_SIG_OPENSSL_SHA512:
			case PHAR_SIG_OPENSSL_SHA256:
			case PHAR_SIG_OPENSSL: {
				uint32_t signature_len;
				char *sig;
				zend_off_t whence;

				/* we store the signature followed by the signature length */
				if (-1 == php_stream_seek(fp, -12, SEEK_CUR)
				|| 4 != php_stream_read(fp, sig_buf, 4)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						spprintf(error, 0, "phar \"%s\" openssl signature length could not be read", fname);
					}
					return FAILURE;
				}

				sig_ptr = sig_buf;
				PHAR_GET_32(sig_ptr, signature_len);
				sig = (char *) emalloc(signature_len);
				whence = signature_len + 4;
				whence = -whence;

				if (-1 == php_stream_seek(fp, whence, SEEK_CUR)
				|| !(end_of_phar = php_stream_tell(fp))
				|| signature_len != php_stream_read(fp, sig, signature_len)) {
					efree(savebuf);
					efree(sig);
					php_stream_close(fp);
					if (error) {
						spprintf(error, 0, "phar \"%s\" openssl signature could not be read", fname);
					}
					return FAILURE;
				}

				if (FAILURE == phar_verify_signature(fp, end_of_phar, sig_flags, sig, signature_len, fname, &signature, &sig_len, error)) {
					efree(savebuf);
					efree(sig);
					php_stream_close(fp);
					if (error) {
						char *save = *error;
						spprintf(error, 0, "phar \"%s\" openssl signature could not be verified: %s", fname, *error);
						efree(save);
					}
					return FAILURE;
				}
				efree(sig);
			}
			break;
			case PHAR_SIG_SHA512: {
				unsigned char digest[64];

				php_stream_seek(fp, -(8 + 64), SEEK_END);
				read_len = php_stream_tell(fp);

				if (php_stream_read(fp, (char*)digest, sizeof(digest)) != sizeof(digest)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						spprintf(error, 0, "phar \"%s\" has a broken signature", fname);
					}
					return FAILURE;
				}

				if (FAILURE == phar_verify_signature(fp, read_len, PHAR_SIG_SHA512, (char *)digest, 64, fname, &signature, &sig_len, error)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						char *save = *error;
						spprintf(error, 0, "phar \"%s\" SHA512 signature could not be verified: %s", fname, *error);
						efree(save);
					}
					return FAILURE;
				}
				break;
			}
			case PHAR_SIG_SHA256: {
				unsigned char digest[32];

				php_stream_seek(fp, -(8 + 32), SEEK_END);
				read_len = php_stream_tell(fp);

				if (php_stream_read(fp, (char*)digest, sizeof(digest)) != sizeof(digest)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						spprintf(error, 0, "phar \"%s\" has a broken signature", fname);
					}
					return FAILURE;
				}

				if (FAILURE == phar_verify_signature(fp, read_len, PHAR_SIG_SHA256, (char *)digest, 32, fname, &signature, &sig_len, error)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						char *save = *error;
						spprintf(error, 0, "phar \"%s\" SHA256 signature could not be verified: %s", fname, *error);
						efree(save);
					}
					return FAILURE;
				}
				break;
			}
			case PHAR_SIG_SHA1: {
				unsigned char digest[20];

				php_stream_seek(fp, -(8 + 20), SEEK_END);
				read_len = php_stream_tell(fp);

				if (php_stream_read(fp, (char*)digest, sizeof(digest)) != sizeof(digest)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						spprintf(error, 0, "phar \"%s\" has a broken signature", fname);
					}
					return FAILURE;
				}

				if (FAILURE == phar_verify_signature(fp, read_len, PHAR_SIG_SHA1, (char *)digest, 20, fname, &signature, &sig_len, error)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						char *save = *error;
						spprintf(error, 0, "phar \"%s\" SHA1 signature could not be verified: %s", fname, *error);
						efree(save);
					}
					return FAILURE;
				}
				break;
			}
			case PHAR_SIG_MD5: {
				unsigned char digest[16];

				php_stream_seek(fp, -(8 + 16), SEEK_END);
				read_len = php_stream_tell(fp);

				if (php_stream_read(fp, (char*)digest, sizeof(digest)) != sizeof(digest)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						spprintf(error, 0, "phar \"%s\" has a broken signature", fname);
					}
					return FAILURE;
				}

				if (FAILURE == phar_verify_signature(fp, read_len, PHAR_SIG_MD5, (char *)digest, 16, fname, &signature, &sig_len, error)) {
					efree(savebuf);
					php_stream_close(fp);
					if (error) {
						char *save = *error;
						spprintf(error, 0, "phar \"%s\" MD5 signature could not be verified: %s", fname, *error);
						efree(save);
					}
					return FAILURE;
				}
				break;
			}
			default:
				efree(savebuf);
				php_stream_close(fp);

				if (error) {
					spprintf(error, 0, "phar \"%s\" has a broken or unsupported signature", fname);
				}
				return FAILURE;
		}
	} else if (PHAR_G(require_hash)) {
		efree(savebuf);
		php_stream_close(fp);

		if (error) {
			spprintf(error, 0, "phar \"%s\" does not have a signature", fname);
		}
		return FAILURE;
	} else {
		sig_flags = 0;
		sig_len = 0;
	}

	/* extract alias */
	SAFE_PHAR_GET_32(buffer, endbuffer, tmp_len);

	if (buffer + tmp_len > endbuffer) {
		MAPPHAR_FAIL("internal corruption of phar \"%s\" (buffer overrun)");
	}

	if (manifest_len < MANIFEST_FIXED_LEN + tmp_len) {
		MAPPHAR_FAIL("internal corruption of phar \"%s\" (truncated manifest header)")
	}

	/* tmp_len = 0 says alias length is 0, which means the alias is not stored in the phar */
	if (tmp_len) {
		/* if the alias is stored we enforce it (implicit overrides explicit) */
		if (alias && alias_len && (alias_len != tmp_len || strncmp(alias, buffer, tmp_len)))
		{
			php_stream_close(fp);

			if (signature) {
				efree(signature);
			}

			if (error) {
				spprintf(error, 0, "cannot load phar \"%s\" with implicit alias \"%.*s\" under different alias \"%s\"", fname, tmp_len, buffer, alias);
			}

			efree(savebuf);
			return FAILURE;
		}

		alias_len = tmp_len;
		alias = buffer;
		buffer += tmp_len;
		register_alias = 1;
	} else if (!alias_len || !alias) {
		/* if we neither have an explicit nor an implicit alias, we use the filename */
		alias = NULL;
		alias_len = 0;
		register_alias = 0;
	} else if (alias_len) {
		register_alias = 1;
		temp_alias = 1;
	}

	/* we have 5 32-bit items plus 1 byte at least */
	if (manifest_count > ((manifest_len - MANIFEST_FIXED_LEN - tmp_len) / (5 * 4 + 1))) {
		/* prevent serious memory issues */
		MAPPHAR_FAIL("internal corruption of phar \"%s\" (too many manifest entries for size of manifest)")
	}

	mydata = pecalloc(1, sizeof(phar_archive_data), PHAR_G(persist));
	mydata->is_persistent = PHAR_G(persist);
	HT_INVALIDATE(&mydata->manifest);
	HT_INVALIDATE(&mydata->mounted_dirs);
	HT_INVALIDATE(&mydata->virtual_dirs);

	/* check whether we have meta data, zero check works regardless of byte order */
	SAFE_PHAR_GET_32(buffer, endbuffer, len);
	if (mydata->is_persistent) {
		if (!len) {
			/* FIXME: not sure why this is needed but removing it breaks tests */
			SAFE_PHAR_GET_32(buffer, endbuffer, len);
		}
	}
	if(len > (size_t)(endbuffer - buffer)) {
		MAPPHAR_FAIL("internal corruption of phar \"%s\" (trying to read past buffer end)");
	}
	/* Don't implicitly call unserialize() on potentially untrusted input unless getMetadata() is called directly. */
	phar_parse_metadata_lazy(buffer, &mydata->metadata_tracker, len, mydata->is_persistent);
	buffer += len;

	/* set up our manifest */
	zend_hash_init(&mydata->manifest, manifest_count,
		zend_get_hash_value, destroy_phar_manifest_entry, (bool)mydata->is_persistent);
	zend_hash_init(&mydata->mounted_dirs, 5,
		zend_get_hash_value, NULL, (bool)mydata->is_persistent);
	zend_hash_init(&mydata->virtual_dirs, manifest_count * 2,
		zend_get_hash_value, NULL, (bool)mydata->is_persistent);
	mydata->fname = pestrndup(fname, fname_len, mydata->is_persistent);
#ifdef PHP_WIN32
	phar_unixify_path_separators(mydata->fname, fname_len);
#endif
	mydata->fname_len = fname_len;
	offset = halt_offset + manifest_len + 4;
	memset(&entry, 0, sizeof(phar_entry_info));
	entry.phar = mydata;
	entry.fp_type = PHAR_FP;
	entry.is_persistent = mydata->is_persistent;

	for (manifest_index = 0; manifest_index < manifest_count; ++manifest_index) {
		if (buffer + 28 > endbuffer) {
			MAPPHAR_FAIL("internal corruption of phar \"%s\" (truncated manifest entry)")
		}

		uint32_t filename_len;
		PHAR_GET_32(buffer, filename_len);

		if (filename_len == 0) {
			MAPPHAR_FAIL("zero-length filename encountered in phar \"%s\"");
		}

		if (entry.is_persistent) {
			entry.manifest_pos = manifest_index;
		}

		if (filename_len > (size_t)(endbuffer - buffer - 24)) {
			MAPPHAR_FAIL("internal corruption of phar \"%s\" (truncated manifest entry)");
		}

		if ((manifest_ver & PHAR_API_VER_MASK) >= PHAR_API_MIN_DIR && buffer[filename_len - 1] == '/') {
			entry.is_dir = 1;
		} else {
			entry.is_dir = 0;
		}

		phar_add_virtual_dirs(mydata, buffer, filename_len);
		const char *filename_raw = buffer;
		buffer += filename_len;
		PHAR_GET_32(buffer, entry.uncompressed_filesize);
		PHAR_GET_32(buffer, entry.timestamp);

		if (offset == halt_offset + manifest_len + 4) {
			mydata->min_timestamp = entry.timestamp;
			mydata->max_timestamp = entry.timestamp;
		} else {
			if (mydata->min_timestamp > entry.timestamp) {
				mydata->min_timestamp = entry.timestamp;
			} else if (mydata->max_timestamp < entry.timestamp) {
				mydata->max_timestamp = entry.timestamp;
			}
		}

		PHAR_GET_32(buffer, entry.compressed_filesize);
		PHAR_GET_32(buffer, entry.crc32);
		PHAR_GET_32(buffer, entry.flags);

		if (entry.is_dir) {
			filename_len--;
			entry.flags |= PHAR_ENT_PERM_DEF_DIR;
		}

		entry.filename = zend_string_init(filename_raw, filename_len, entry.is_persistent);
		if (entry.is_persistent) {
			GC_MAKE_PERSISTENT_LOCAL(entry.filename);
		}

		PHAR_GET_32(buffer, len);
		if (len > (size_t)(endbuffer - buffer)) {
			zend_string_free(entry.filename);
			MAPPHAR_FAIL("internal corruption of phar \"%s\" (truncated manifest entry)");
		}
		/* Don't implicitly call unserialize() on potentially untrusted input unless getMetadata() is called directly. */
		/* The same local variable entry is reused in a loop, so reset the state before reading data. */
		ZVAL_UNDEF(&entry.metadata_tracker.val);
		entry.metadata_tracker.str = NULL;
		phar_parse_metadata_lazy(buffer, &entry.metadata_tracker, len, entry.is_persistent);
		buffer += len;

		entry.offset = entry.offset_abs = offset;
		offset += entry.compressed_filesize;

		switch (entry.flags & PHAR_ENT_COMPRESSION_MASK) {
			case PHAR_ENT_COMPRESSED_GZ:
				if (!PHAR_G(has_zlib)) {
					phar_metadata_tracker_free(&entry.metadata_tracker, entry.is_persistent);
					zend_string_free(entry.filename);
					MAPPHAR_FAIL("zlib extension is required for gz compressed .phar file \"%s\"");
				}
				break;
			case PHAR_ENT_COMPRESSED_BZ2:
				if (!PHAR_G(has_bz2)) {
					phar_metadata_tracker_free(&entry.metadata_tracker, entry.is_persistent);
					zend_string_free(entry.filename);
					MAPPHAR_FAIL("bz2 extension is required for bzip2 compressed .phar file \"%s\"");
				}
				break;
			default:
				if (entry.uncompressed_filesize != entry.compressed_filesize) {
					phar_metadata_tracker_free(&entry.metadata_tracker, entry.is_persistent);
					zend_string_free(entry.filename);
					MAPPHAR_FAIL("internal corruption of phar \"%s\" (compressed and uncompressed size does not match for uncompressed entry)");
				}
				break;
		}

		manifest_flags |= (entry.flags & PHAR_ENT_COMPRESSION_MASK);
