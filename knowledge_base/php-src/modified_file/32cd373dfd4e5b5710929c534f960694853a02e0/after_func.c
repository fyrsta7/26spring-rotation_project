		/*
		  Because RSA_PKCS1_OAEP_PADDING is used there is a restriction on the passwd_len.
		  RSA_PKCS1_OAEP_PADDING is recommended for new applications. See more here:
		  http://www.openssl.org/docs/crypto/RSA_public_encrypt.html
		*/
		if ((size_t) server_public_key_len - 41 <= passwd_len) {
			/* password message is to long */
			SET_CLIENT_ERROR(conn->error_info, CR_UNKNOWN_ERROR, UNKNOWN_SQLSTATE, "password is too long");
			DBG_ERR("password is too long");
			DBG_RETURN(0);
		}

		*crypted = emalloc(server_public_key_len);
		RSA_public_encrypt(passwd_len + 1, (zend_uchar *) xor_str, *crypted, server_public_key, RSA_PKCS1_OAEP_PADDING);
		DBG_RETURN(server_public_key_len);
	}
	DBG_RETURN(0);
}
/* }}} */

static int is_secure_transport(MYSQLND_CONN_DATA *conn) {
	if (conn->vio->data->ssl) {
		return 1;
	}

	return strcmp(conn->vio->data->stream->ops->label, "unix_socket") == 0;
}

/* {{{ mysqlnd_caching_sha2_handle_server_response */
static enum_func_status
mysqlnd_caching_sha2_handle_server_response(struct st_mysqlnd_authentication_plugin *self,
		MYSQLND_CONN_DATA * conn,
		const zend_uchar * auth_plugin_data, const size_t auth_plugin_data_len,
		const char * const passwd,
		const size_t passwd_len,
		char **new_auth_protocol, size_t *new_auth_protocol_len,
		zend_uchar **new_auth_protocol_data, size_t *new_auth_protocol_data_len
		)
{
	DBG_ENTER("mysqlnd_caching_sha2_handle_server_response");
	MYSQLND_PACKET_CACHED_SHA2_RESULT result_packet;

	if (passwd_len == 0) {
		DBG_INF("empty password fast path");
		DBG_RETURN(PASS);
	}

	conn->payload_decoder_factory->m.init_cached_sha2_result_packet(&result_packet);
	if (FAIL == PACKET_READ(conn, &result_packet)) {
		DBG_RETURN(PASS);
	}

	switch (result_packet.response_code) {
		case 0xFF:
			if (result_packet.sqlstate[0]) {
				strlcpy(conn->error_info->sqlstate, result_packet.sqlstate, sizeof(conn->error_info->sqlstate));
				DBG_ERR_FMT("ERROR:%u [SQLSTATE:%s] %s", result_packet.error_no, result_packet.sqlstate, result_packet.error);
			}
			SET_CLIENT_ERROR(conn->error_info, result_packet.error_no, UNKNOWN_SQLSTATE, result_packet.error);
			DBG_RETURN(FAIL);
		case 0xFE:
			DBG_INF("auth switch response");
			*new_auth_protocol = result_packet.new_auth_protocol;
