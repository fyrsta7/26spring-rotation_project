)
{
	dom_lexbor_libxml2_bridge_application_data *application_data = ctx->application_data;
	application_data->current_input_length = input_buffer_length;
	lexbor_status_t lexbor_status = lxb_html_document_parse_chunk(document, encoding_output, encoded_length);
	if (UNEXPECTED(lexbor_status != LXB_STATUS_OK)) {
		return false;
	}
	if (ctx->tokenizer_error_reporter || ctx->tree_error_reporter) {
		lexbor_libxml2_bridge_report_errors(ctx, parser, encoding_output, application_data->current_total_offset, tokenizer_error_offset, tree_error_offset);
		dom_find_line_and_column_using_cache(application_data, &application_data->cache_tokenizer, application_data->current_total_offset + input_buffer_length);
	}
	application_data->current_total_offset += input_buffer_length;
	application_data->cache_tokenizer.last_offset = 0;
	return true;
}

static bool dom_decode_encode_fast_path(
	lexbor_libxml2_bridge_parse_context *ctx,
	lxb_html_document_t *document,
	lxb_html_parser_t *parser,
	const lxb_char_t **buf_ref_ref,
	const lxb_char_t *buf_end,
	dom_decoding_encoding_ctx *decoding_encoding_ctx,
	size_t *tokenizer_error_offset,
	size_t *tree_error_offset
)
{
	const lxb_char_t *buf_ref = *buf_ref_ref;
	const lxb_char_t *last_output = buf_ref;
	while (buf_ref != buf_end) {
		const lxb_char_t *buf_ref_backup = buf_ref;
		/* Fast path converts non-validated UTF-8 -> validated UTF-8 */
		lxb_codepoint_t codepoint = lxb_encoding_decode_utf_8_single(&decoding_encoding_ctx->decode, &buf_ref, buf_end);
		if (UNEXPECTED(codepoint > LXB_ENCODING_MAX_CODEPOINT)) {
			size_t skip = buf_ref - buf_ref_backup; /* Skip invalid data, it's replaced by the UTF-8 replacement bytes */
			if (!dom_process_parse_chunk(
				ctx,
				document,
				parser,
				buf_ref - last_output - skip,
				last_output,
				buf_ref - last_output,
				tokenizer_error_offset,
				tree_error_offset
			)) {
				goto fail_oom;
			}
			if (!dom_process_parse_chunk(
				ctx,
				document,
				parser,
				LXB_ENCODING_REPLACEMENT_SIZE,
				LXB_ENCODING_REPLACEMENT_BYTES,
				0,
				tokenizer_error_offset,
				tree_error_offset
			)) {
				goto fail_oom;
			}
			last_output = buf_ref;
		}
	}
	if (buf_ref != last_output
		&& !dom_process_parse_chunk(
