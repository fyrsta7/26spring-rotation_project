			field = form->field[j];

			if (0 == innobase_strcasecmp(
					field->field_name,
					key_part->field->field_name)) {
				/* Found the corresponding column */

				break;
			}
		}

		ut_a(j < form->s->fields);

		col_type = get_innobase_type_from_mysql_type(
					&is_unsigned, key_part->field);

		if (DATA_BLOB == col_type
			|| (key_part->length < field->pack_length()
				&& field->type() != MYSQL_TYPE_VARCHAR)
			|| (field->type() == MYSQL_TYPE_VARCHAR
				&& key_part->length < field->pack_length()
				- ((Field_varstring*)field)->length_bytes)) {

			prefix_len = key_part->length;

			if (col_type == DATA_INT
				|| col_type == DATA_FLOAT
				|| col_type == DATA_DOUBLE
				|| col_type == DATA_DECIMAL) {
				sql_print_error(
					"MySQL is trying to create a column "
					"prefix index field, on an "
					"inappropriate data type. Table "
					"name %s, column name %s.",
					table_name,
					key_part->field->field_name);

				prefix_len = 0;
			}
		} else {
			prefix_len = 0;
		}

		field_lengths[i] = key_part->length;

		dict_mem_index_add_field(index,
			(char*) key_part->field->field_name, prefix_len);
	}

	/* Even though we've defined max_supported_key_part_length, we
	still do our own checking using field_lengths to be absolutely
	sure we don't create too long indexes. */
	error = row_create_index_for_mysql(index, trx, field_lengths);

	error = convert_error_code_to_mysql(error, flags, NULL);

	my_free(field_lengths);

	DBUG_RETURN(error);
}

/*****************************************************************//**
Creates an index to an InnoDB table when the user has defined no
primary index. */
static
int
create_clustered_index_when_no_primary(
/*===================================*/
	trx_t*		trx,		/*!< in: InnoDB transaction handle */
	ulint		flags,		/*!< in: InnoDB table flags */
	const char*	table_name)	/*!< in: table name */
{
	dict_index_t*	index;
	int		error;

	/* We pass 0 as the space id, and determine at a lower level the space
	id where to store the table */
	index = dict_mem_index_create(table_name,
				      innobase_index_reserve_name,
				      0, DICT_CLUSTERED, 0);

	error = row_create_index_for_mysql(index, trx, NULL);

	error = convert_error_code_to_mysql(error, flags, NULL);

	return(error);
}

/*****************************************************************//**
Return a display name for the row format
@return row format name */
UNIV_INTERN
const char*
get_row_format_name(
/*================*/
	enum row_type	row_format)		/*!< in: Row Format */
{
	switch (row_format) {
	case ROW_TYPE_COMPACT:
		return("COMPACT");
	case ROW_TYPE_COMPRESSED:
		return("COMPRESSED");
	case ROW_TYPE_DYNAMIC:
		return("DYNAMIC");
	case ROW_TYPE_REDUNDANT:
		return("REDUNDANT");
	case ROW_TYPE_DEFAULT:
		return("DEFAULT");
	case ROW_TYPE_FIXED:
		return("FIXED");
	case ROW_TYPE_PAGE:
	case ROW_TYPE_NOT_USED:
		break;
	}
	return("NOT USED");
}

/** If file-per-table is missing, issue warning and set ret false */
#define CHECK_ERROR_ROW_TYPE_NEEDS_FILE_PER_TABLE		\
	if (!srv_file_per_table) {				\
		push_warning_printf(				\
			thd, MYSQL_ERROR::WARN_LEVEL_WARN,	\
			ER_ILLEGAL_HA_CREATE_OPTION,		\
			"InnoDB: ROW_FORMAT=%s requires"	\
			" innodb_file_per_table.",		\
			get_row_format_name(row_format));	\
		ret = FALSE;					\
	}

/** If file-format is Antelope, issue warning and set ret false */
#define CHECK_ERROR_ROW_TYPE_NEEDS_GT_ANTELOPE			\
	if (srv_file_format < DICT_TF_FORMAT_ZIP) {		\
		push_warning_printf(				\
			thd, MYSQL_ERROR::WARN_LEVEL_WARN,	\
			ER_ILLEGAL_HA_CREATE_OPTION,		\
			"InnoDB: ROW_FORMAT=%s requires"	\
			" innodb_file_format > Antelope.",	\
			get_row_format_name(row_format));	\
		ret = FALSE;					\
	}


/*****************************************************************//**
Validates the create options. We may build on this function
in future. For now, it checks two specifiers:
KEY_BLOCK_SIZE and ROW_FORMAT
If innodb_strict_mode is not set then this function is a no-op
@return	TRUE if valid. */
static
ibool
create_options_are_valid(
/*=====================*/
	THD*		thd,		/*!< in: connection thread. */
	TABLE*		form,		/*!< in: information on table
					columns and indexes */
	HA_CREATE_INFO*	create_info)	/*!< in: create info. */
{
	ibool	kbs_specified	= FALSE;
	ibool	ret		= TRUE;
	enum row_type	row_format	= form->s->row_type;

	ut_ad(thd != NULL);

	/* If innodb_strict_mode is not set don't do any validation. */
	if (!(THDVAR(thd, strict_mode))) {
		return(TRUE);
	}

	ut_ad(form != NULL);
	ut_ad(create_info != NULL);

	/* First check if a non-zero KEY_BLOCK_SIZE was specified. */
	if (create_info->key_block_size) {
		kbs_specified = TRUE;
		switch (create_info->key_block_size) {
		case 1:
		case 2:
		case 4:
		case 8:
		case 16:
			/* Valid KEY_BLOCK_SIZE, check its dependencies. */
			if (!srv_file_per_table) {
				push_warning(
					thd, MYSQL_ERROR::WARN_LEVEL_WARN,
					ER_ILLEGAL_HA_CREATE_OPTION,
					"InnoDB: KEY_BLOCK_SIZE requires"
					" innodb_file_per_table.");
				ret = FALSE;
			}
			if (srv_file_format < DICT_TF_FORMAT_ZIP) {
				push_warning(
					thd, MYSQL_ERROR::WARN_LEVEL_WARN,
					ER_ILLEGAL_HA_CREATE_OPTION,
					"InnoDB: KEY_BLOCK_SIZE requires"
					" innodb_file_format > Antelope.");
					ret = FALSE;
			}
			break;
		default:
			push_warning_printf(
				thd, MYSQL_ERROR::WARN_LEVEL_WARN,
				ER_ILLEGAL_HA_CREATE_OPTION,
				"InnoDB: invalid KEY_BLOCK_SIZE = %lu."
				" Valid values are [1, 2, 4, 8, 16]",
				create_info->key_block_size);
			ret = FALSE;
			break;
		}
	}
	
	/* Check for a valid Innodb ROW_FORMAT specifier and
	other incompatibilities. */
	switch (row_format) {
	case ROW_TYPE_COMPRESSED:
		CHECK_ERROR_ROW_TYPE_NEEDS_FILE_PER_TABLE;
		CHECK_ERROR_ROW_TYPE_NEEDS_GT_ANTELOPE;
		break;
	case ROW_TYPE_DYNAMIC:
		CHECK_ERROR_ROW_TYPE_NEEDS_FILE_PER_TABLE;
		CHECK_ERROR_ROW_TYPE_NEEDS_GT_ANTELOPE;
		/* fall through since dynamic also shuns KBS */
	case ROW_TYPE_COMPACT:
	case ROW_TYPE_REDUNDANT:
		if (kbs_specified) {
			push_warning_printf(
				thd, MYSQL_ERROR::WARN_LEVEL_WARN,
				ER_ILLEGAL_HA_CREATE_OPTION,
				"InnoDB: cannot specify ROW_FORMAT = %s"
				" with KEY_BLOCK_SIZE.",
				get_row_format_name(row_format));
			ret = FALSE;
		}
		break;
	case ROW_TYPE_DEFAULT:
		break;
	case ROW_TYPE_FIXED:
	case ROW_TYPE_PAGE:
	case ROW_TYPE_NOT_USED:
		push_warning(
			thd, MYSQL_ERROR::WARN_LEVEL_WARN,
			ER_ILLEGAL_HA_CREATE_OPTION,		\
			"InnoDB: invalid ROW_FORMAT specifier.");
		ret = FALSE;
		break;
	}

	return(ret);
}

/*****************************************************************//**
Update create_info.  Used in SHOW CREATE TABLE et al. */
