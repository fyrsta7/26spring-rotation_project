		memset(*p, 0, null_count);
		*p += null_count;
	}

/* 1. Store type information */
	/*
	  check if need to send the types even if stmt->send_types_to_server is 0. This is because
	  if we send "i" (42) then the type will be int and the server will expect int. However, if next
	  time we try to send > LONG_MAX, the conversion to string will send a string and the server
	  won't expect it and interpret the value as 0. Thus we need to resend the types, if any such values
	  occur, and force resend for the next execution.
	*/
	if (FAIL == mysqlnd_stmt_execute_prepare_param_types(stmt, &copies, &resend_types_next_time)) {
		goto end;
	}

	int1store(*p, stmt->send_types_to_server);
	(*p)++;

	if (stmt->send_types_to_server) {
		if (FAIL == mysqlnd_stmt_execute_check_n_enlarge_buffer(buf, p, buf_len, provided_buffer, stmt->param_count * 2)) {
			SET_OOM_ERROR(stmt->error_info);
			goto end;
		}
		mysqlnd_stmt_execute_store_types(stmt, copies, p);
	}

	stmt->send_types_to_server = resend_types_next_time;

/* 2. Store data */
	/* 2.1 Calculate how much space we need */
	if (FAIL == mysqlnd_stmt_execute_calculate_param_values_size(stmt, &copies, &data_size)) {
