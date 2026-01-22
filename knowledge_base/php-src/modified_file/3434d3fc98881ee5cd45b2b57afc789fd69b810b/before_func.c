		memmove(new_end, src, len*sizeof(zend_op));

		new_end += len;
	}
	block->len = new_end - block->start_opline;
}

static void zend_optimize_block(zend_code_block *block, zend_op_array *op_array, char *used_ext TSRMLS_DC)
{
	zend_op *opline = block->start_opline;
	zend_op *end, *last_op = NULL;
	zend_op **Tsource = NULL;

	print_block(block, op_array->opcodes, "Opt ");

	/* remove leading NOPs */
	while (block->len > 0 && block->start_opline->opcode == ZEND_NOP) {
		if (block->len == 1) {
			/* this block is all NOPs, join with following block */
			if (block->follow_to) {
				delete_code_block(block);
			}
			return;
		}
		block->start_opline++;
		block->start_opline_no++;
		block->len--;
	}

	/* we track data dependencies only insight a single basic block */
	if (op_array->T) {
		Tsource = ecalloc(op_array->last_var + op_array->T, sizeof(zend_op *));
	}
	opline = block->start_opline;
	end = opline + block->len;
	while ((op_array->T) && (opline < end)) {
		/* strip X = QM_ASSIGN(const) */
		if (ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
			VAR_SOURCE(opline->op1) &&
			VAR_SOURCE(opline->op1)->opcode == ZEND_QM_ASSIGN &&
			ZEND_OP1_TYPE(VAR_SOURCE(opline->op1)) == IS_CONST &&
			opline->opcode != ZEND_CASE &&         /* CASE _always_ expects variable */
			opline->opcode != ZEND_FETCH_DIM_TMP_VAR &&   /* in 5.1, FETCH_DIM_TMP_VAR expects T */
			opline->opcode != ZEND_FE_RESET &&
			opline->opcode != ZEND_FREE
			) {
			zend_op *src = VAR_SOURCE(opline->op1);
			zval c = ZEND_OP1_LITERAL(src);
			VAR_UNSET(opline->op1);
			zval_copy_ctor(&c);
			update_op1_const(op_array, opline, &c TSRMLS_CC);
			literal_dtor(&ZEND_OP1_LITERAL(src));
			MAKE_NOP(src);
		}

		/* T = QM_ASSIGN(C), F(T) => NOP, F(C) */
		if (ZEND_OP2_TYPE(opline) == IS_TMP_VAR &&
			VAR_SOURCE(opline->op2) &&
			VAR_SOURCE(opline->op2)->opcode == ZEND_QM_ASSIGN &&
			ZEND_OP1_TYPE(VAR_SOURCE(opline->op2)) == IS_CONST) {
			zend_op *src = VAR_SOURCE(opline->op2);
			zval c = ZEND_OP1_LITERAL(src);
			VAR_UNSET(opline->op2);
			zval_copy_ctor(&c);
			update_op2_const(op_array, opline, &c TSRMLS_CC);
			literal_dtor(&ZEND_OP1_LITERAL(src));
			MAKE_NOP(src);
		}

		/* T = PRINT(X), F(T) => ECHO(X), F(1) */
		if (ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
			VAR_SOURCE(opline->op1) &&
			VAR_SOURCE(opline->op1)->opcode == ZEND_PRINT &&
			opline->opcode != ZEND_CASE && opline->opcode != ZEND_FREE) {
			ZEND_OP1_TYPE(opline) = IS_CONST;
			LITERAL_LONG(opline->op1, 1);
		}

		if (ZEND_OP2_TYPE(opline) == IS_TMP_VAR &&
			VAR_SOURCE(opline->op2) &&
			VAR_SOURCE(opline->op2)->opcode == ZEND_PRINT) {
			ZEND_OP2_TYPE(opline) = IS_CONST;
			LITERAL_LONG(opline->op2, 1);
		}

		/* T = CAST(X, String), ECHO(T) => NOP, ECHO(X) */
		if ((opline->opcode == ZEND_ECHO || opline->opcode == ZEND_PRINT) &&
			ZEND_OP1_TYPE(opline) & (IS_TMP_VAR|IS_VAR) &&
			VAR_SOURCE(opline->op1) &&
			VAR_SOURCE(opline->op1)->opcode == ZEND_CAST &&
			VAR_SOURCE(opline->op1)->extended_value == IS_STRING) {
			zend_op *src = VAR_SOURCE(opline->op1);
			COPY_NODE(opline->op1, src->op1);
			MAKE_NOP(src);
		}

		/* T = PRINT(X), FREE(T) => ECHO(X) */
		if (opline->opcode == ZEND_FREE &&
			ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
			VAR_SOURCE(opline->op1)) {
			zend_op *src = VAR_SOURCE(opline->op1);
			if (src->opcode == ZEND_PRINT) {
				src->opcode = ZEND_ECHO;
				ZEND_RESULT_TYPE(src) = IS_UNUSED;
				MAKE_NOP(opline);
			}
		}

       /* T = BOOL(X), FREE(T) => NOP */
		if (opline->opcode == ZEND_FREE &&
			ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
			VAR_SOURCE(opline->op1)) {
			zend_op *src = VAR_SOURCE(opline->op1);
			if (src->opcode == ZEND_BOOL) {
				if (ZEND_OP1_TYPE(src) == IS_CONST) {
					literal_dtor(&ZEND_OP1_LITERAL(src));
				}
				MAKE_NOP(src);
				MAKE_NOP(opline);
			}
		}

#if 0
		/* pre-evaluate functions:
		   constant(x)
		   defined(x)
		   function_exists(x)
		   extension_loaded(x)
		   BAD: interacts badly with Accelerator
		*/
		if((ZEND_OP1_TYPE(opline) & IS_VAR) &&
		   VAR_SOURCE(opline->op1) && VAR_SOURCE(opline->op1)->opcode == ZEND_DO_CF_FCALL &&
		   VAR_SOURCE(opline->op1)->extended_value == 1) {
			zend_op *fcall = VAR_SOURCE(opline->op1);
			zend_op *sv = fcall-1;
			if(sv >= block->start_opline && sv->opcode == ZEND_SEND_VAL &&
			   ZEND_OP1_TYPE(sv) == IS_CONST && Z_TYPE(OPLINE_OP1_LITERAL(sv)) == IS_STRING &&
			   Z_LVAL(OPLINE_OP2_LITERAL(sv)) == 1
			   ) {
				zval *arg = &OPLINE_OP1_LITERAL(sv);
				char *fname = FUNCTION_CACHE->funcs[Z_LVAL(ZEND_OP1_LITERAL(fcall))].function_name;
				int flen = FUNCTION_CACHE->funcs[Z_LVAL(ZEND_OP1_LITERAL(fcall))].name_len;
				if(flen == sizeof("defined")-1 && zend_binary_strcasecmp(fname, flen, "defined", sizeof("defined")-1) == 0) {
					zval c;
					if(zend_get_persistent_constant(Z_STR_P(arg), &c, 0 TSRMLS_CC ELS_CC) != 0) {
						literal_dtor(arg);
						MAKE_NOP(sv);
						MAKE_NOP(fcall);
						LITERAL_BOOL(opline->op1, 1);
						ZEND_OP1_TYPE(opline) = IS_CONST;
					}
				} else if((flen == sizeof("function_exists")-1 && zend_binary_strcasecmp(fname, flen, "function_exists", sizeof("function_exists")-1) == 0) ||
						  (flen == sizeof("is_callable")-1 && zend_binary_strcasecmp(fname, flen, "is_callable", sizeof("is_callable")-1) == 0)
						  ) {
					zend_function *function;
					if(zend_hash_find(EG(function_table), Z_STRVAL_P(arg), Z_STRLEN_P(arg)+1, (void **)&function) == SUCCESS) {
						literal_dtor(arg);
						MAKE_NOP(sv);
						MAKE_NOP(fcall);
						LITERAL_BOOL(opline->op1, 1);
						ZEND_OP1_TYPE(opline) = IS_CONST;
					}
				} else if(flen == sizeof("constant")-1 && zend_binary_strcasecmp(fname, flen, "constant", sizeof("constant")-1) == 0) {
					zval c;
					if(zend_get_persistent_constant(Z_STR_P(arg), &c, 1 TSRMLS_CC ELS_CC) != 0) {
						literal_dtor(arg);
						MAKE_NOP(sv);
						MAKE_NOP(fcall);
						ZEND_OP1_LITERAL(opline) = zend_optimizer_add_literal(op_array, &c TSRMLS_CC);
						/* no copy ctor - get already copied it */
						ZEND_OP1_TYPE(opline) = IS_CONST;
					}
				} else if(flen == sizeof("extension_loaded")-1 && zend_binary_strcasecmp(fname, flen, "extension_loaded", sizeof("extension_loaded")-1) == 0) {
					if(zend_hash_exists(&module_registry, Z_STRVAL_P(arg), Z_STRLEN_P(arg)+1)) {
						literal_dtor(arg);
						MAKE_NOP(sv);
						MAKE_NOP(fcall);
						LITERAL_BOOL(opline->op1, 1);
						ZEND_OP1_TYPE(opline) = IS_CONST;
					}
				}
			}
		}
#endif

        /* IS_EQ(TRUE, X)      => BOOL(X)
         * IS_EQ(FALSE, X)     => BOOL_NOT(X)
         * IS_NOT_EQ(TRUE, X)  => BOOL_NOT(X)
         * IS_NOT_EQ(FALSE, X) => BOOL(X)
         */
		if (opline->opcode == ZEND_IS_EQUAL ||
			opline->opcode == ZEND_IS_NOT_EQUAL) {
			if (ZEND_OP1_TYPE(opline) == IS_CONST &&
				Z_TYPE(ZEND_OP1_LITERAL(opline)) == IS_BOOL) {
				opline->opcode =
					((opline->opcode == ZEND_IS_EQUAL) == Z_LVAL(ZEND_OP1_LITERAL(opline)))?
					ZEND_BOOL : ZEND_BOOL_NOT;
				COPY_NODE(opline->op1, opline->op2);
				SET_UNUSED(opline->op2);
			} else if (ZEND_OP2_TYPE(opline) == IS_CONST &&
					   Z_TYPE(ZEND_OP2_LITERAL(opline)) == IS_BOOL) {
				opline->opcode =
					((opline->opcode == ZEND_IS_EQUAL) == Z_LVAL(ZEND_OP2_LITERAL(opline)))?
					ZEND_BOOL : ZEND_BOOL_NOT;
				SET_UNUSED(opline->op2);
			}
		}

		if ((opline->opcode == ZEND_BOOL ||
			opline->opcode == ZEND_BOOL_NOT ||
			opline->opcode == ZEND_JMPZ ||
			opline->opcode == ZEND_JMPNZ ||
			opline->opcode == ZEND_JMPZNZ) &&
			ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
			VAR_SOURCE(opline->op1) != NULL &&
			!used_ext[VAR_NUM(ZEND_OP1(opline).var)] &&
			VAR_SOURCE(opline->op1)->opcode == ZEND_BOOL_NOT) {
			/* T = BOOL_NOT(X) + JMPZ(T) -> NOP, JMPNZ(X) */
			zend_op *src = VAR_SOURCE(opline->op1);

			COPY_NODE(opline->op1, src->op1);

			switch (opline->opcode) {
				case ZEND_BOOL:
					/* T = BOOL_NOT(X) + BOOL(T) -> NOP, BOOL_NOT(X) */
					opline->opcode = ZEND_BOOL_NOT;
					break;
				case ZEND_BOOL_NOT:
					/* T = BOOL_NOT(X) + BOOL_BOOL(T) -> NOP, BOOL(X) */
					opline->opcode = ZEND_BOOL;
					break;
				case ZEND_JMPZ:
					/* T = BOOL_NOT(X) + JMPZ(T,L) -> NOP, JMPNZ(X,L) */
					opline->opcode = ZEND_JMPNZ;
					break;
				case ZEND_JMPNZ:
					/* T = BOOL_NOT(X) + JMPNZ(T,L) -> NOP, JMPZ(X,L) */
					opline->opcode = ZEND_JMPZ;
					break;
				case ZEND_JMPZNZ:
				{
					/* T = BOOL_NOT(X) + JMPZNZ(T,L1,L2) -> NOP, JMPZNZ(X,L2,L1) */
					int op_t;
					zend_code_block *op_b;

					op_t = opline->extended_value;
					opline->extended_value = ZEND_OP2(opline).opline_num;
					ZEND_OP2(opline).opline_num = op_t;

					op_b = block->ext_to;
					block->ext_to = block->op2_to;
					block->op2_to = op_b;
				}
				break;
			}

			VAR_UNSET(opline->op1);
			MAKE_NOP(src);
			continue;
		} else
#if 0
		/* T = BOOL_NOT(X) + T = JMPZ_EX(T, X) -> T = BOOL_NOT(X), JMPNZ(X) */
		if(0 && (opline->opcode == ZEND_JMPZ_EX ||
			opline->opcode == ZEND_JMPNZ_EX) &&
		   ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
		   VAR_SOURCE(opline->op1) != NULL &&
		   VAR_SOURCE(opline->op1)->opcode == ZEND_BOOL_NOT &&
		   ZEND_OP1(opline).var == ZEND_RESULT(opline).var
		   ) {
			zend_op *src = VAR_SOURCE(opline->op1);
			if(opline->opcode == ZEND_JMPZ_EX) {
				opline->opcode = ZEND_JMPNZ;
			} else {
				opline->opcode = ZEND_JMPZ;
			}
			COPY_NODE(opline->op1, src->op1);
			SET_UNUSED(opline->result);
			continue;
		} else
#endif
		/* T = BOOL(X) + JMPZ(T) -> NOP, JMPZ(X) */
		if ((opline->opcode == ZEND_BOOL ||
			opline->opcode == ZEND_BOOL_NOT ||
			opline->opcode == ZEND_JMPZ ||
			opline->opcode == ZEND_JMPZ_EX ||
			opline->opcode == ZEND_JMPNZ_EX ||
			opline->opcode == ZEND_JMPNZ ||
			opline->opcode == ZEND_JMPZNZ) &&
			ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
			VAR_SOURCE(opline->op1) != NULL &&
			(!used_ext[VAR_NUM(ZEND_OP1(opline).var)] ||
			(ZEND_RESULT_TYPE(opline) == IS_TMP_VAR &&
			 ZEND_RESULT(opline).var == ZEND_OP1(opline).var)) &&
			(VAR_SOURCE(opline->op1)->opcode == ZEND_BOOL ||
			VAR_SOURCE(opline->op1)->opcode == ZEND_QM_ASSIGN)) {
			zend_op *src = VAR_SOURCE(opline->op1);
			COPY_NODE(opline->op1, src->op1);

			VAR_UNSET(opline->op1);
			MAKE_NOP(src);
			continue;
		} else if (last_op && opline->opcode == ZEND_ECHO &&
				  last_op->opcode == ZEND_ECHO &&
				  ZEND_OP1_TYPE(opline) == IS_CONST &&
				  Z_TYPE(ZEND_OP1_LITERAL(opline)) != IS_DOUBLE &&
				  ZEND_OP1_TYPE(last_op) == IS_CONST &&
				  Z_TYPE(ZEND_OP1_LITERAL(last_op)) != IS_DOUBLE) {
			/* compress consecutive ECHO's.
			 * Float to string conversion may be affected by current
			 * locale setting.
			 */
			int l, old_len;

			if (Z_TYPE(ZEND_OP1_LITERAL(opline)) != IS_STRING) {
				convert_to_string_safe(&ZEND_OP1_LITERAL(opline));
			}
			if (Z_TYPE(ZEND_OP1_LITERAL(last_op)) != IS_STRING) {
				convert_to_string_safe(&ZEND_OP1_LITERAL(last_op));
			}
			old_len = Z_STRLEN(ZEND_OP1_LITERAL(last_op));
			l = old_len + Z_STRLEN(ZEND_OP1_LITERAL(opline));
			if (IS_INTERNED(Z_STR(ZEND_OP1_LITERAL(last_op)))) {
				zend_string *tmp = STR_ALLOC(l, 0);
				memcpy(tmp->val, Z_STRVAL(ZEND_OP1_LITERAL(last_op)), old_len);
				Z_STR(ZEND_OP1_LITERAL(last_op)) = tmp;
			} else {
				Z_STR(ZEND_OP1_LITERAL(last_op)) = STR_REALLOC(Z_STR(ZEND_OP1_LITERAL(last_op)), l, 0);
			}
			Z_TYPE_INFO(ZEND_OP1_LITERAL(last_op)) = IS_STRING_EX;
			memcpy(Z_STRVAL(ZEND_OP1_LITERAL(last_op)) + old_len, Z_STRVAL(ZEND_OP1_LITERAL(opline)), Z_STRLEN(ZEND_OP1_LITERAL(opline)));
			Z_STRVAL(ZEND_OP1_LITERAL(last_op))[l] = '\0';
			zval_dtor(&ZEND_OP1_LITERAL(opline));
#if ZEND_EXTENSION_API_NO > PHP_5_3_X_API_NO
			Z_STR(ZEND_OP1_LITERAL(opline)) = zend_new_interned_string(Z_STR(ZEND_OP1_LITERAL(last_op)) TSRMLS_CC);
			if (IS_INTERNED(Z_STR(ZEND_OP1_LITERAL(opline)))) {
				Z_TYPE_FLAGS(ZEND_OP1_LITERAL(opline)) &= ~ (IS_TYPE_REFCOUNTED | IS_TYPE_COPYABLE);
			}
			ZVAL_NULL(&ZEND_OP1_LITERAL(last_op));
#else
			Z_STR(ZEND_OP1_LITERAL(opline)) = Z_STR(ZEND_OP1_LITERAL(last_op));
#endif
			MAKE_NOP(last_op);
		} else if (opline->opcode == ZEND_CONCAT &&
				  ZEND_OP2_TYPE(opline) == IS_CONST &&
				  ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
				  VAR_SOURCE(opline->op1) &&
				  (VAR_SOURCE(opline->op1)->opcode == ZEND_CONCAT ||
				   VAR_SOURCE(opline->op1)->opcode == ZEND_ADD_STRING) &&
				  ZEND_OP2_TYPE(VAR_SOURCE(opline->op1)) == IS_CONST &&
				  ZEND_RESULT(VAR_SOURCE(opline->op1)).var == ZEND_OP1(opline).var) {
			/* compress consecutive CONCATs */
			zend_op *src = VAR_SOURCE(opline->op1);
			int l, old_len;

			if (Z_TYPE(ZEND_OP2_LITERAL(opline)) != IS_STRING) {
				convert_to_string_safe(&ZEND_OP2_LITERAL(opline));
			}
			if (Z_TYPE(ZEND_OP2_LITERAL(src)) != IS_STRING) {
				convert_to_string_safe(&ZEND_OP2_LITERAL(src));
			}

			VAR_UNSET(opline->op1);
			if (ZEND_OP1_TYPE(src) == IS_UNUSED) {
				/* 5.3 may use IS_UNUSED as first argument to ZEND_ADD_... */
				opline->opcode = ZEND_ADD_STRING;
			}
			COPY_NODE(opline->op1, src->op1);
			old_len = Z_STRLEN(ZEND_OP2_LITERAL(src));
			l = old_len + Z_STRLEN(ZEND_OP2_LITERAL(opline));
			if (IS_INTERNED(Z_STR(ZEND_OP2_LITERAL(src)))) {
				zend_string *tmp = STR_ALLOC(l, 0);
				memcpy(tmp->val, Z_STRVAL(ZEND_OP2_LITERAL(src)), old_len);
				Z_STR(ZEND_OP2_LITERAL(last_op)) = tmp;
			} else {
				Z_STR(ZEND_OP2_LITERAL(src)) = STR_REALLOC(Z_STR(ZEND_OP2_LITERAL(src)), l, 0);
			}
			Z_TYPE_INFO(ZEND_OP2_LITERAL(last_op)) = IS_STRING_EX;
			memcpy(Z_STRVAL(ZEND_OP2_LITERAL(src)) + old_len, Z_STRVAL(ZEND_OP2_LITERAL(opline)), Z_STRLEN(ZEND_OP2_LITERAL(opline)));
			Z_STRVAL(ZEND_OP2_LITERAL(src))[l] = '\0';
			STR_RELEASE(Z_STR(ZEND_OP2_LITERAL(opline)));
#if ZEND_EXTENSION_API_NO > PHP_5_3_X_API_NO
			Z_STR(ZEND_OP2_LITERAL(opline)) = zend_new_interned_string(Z_STR(ZEND_OP2_LITERAL(src)) TSRMLS_CC);
			if (IS_INTERNED(Z_STR(ZEND_OP2_LITERAL(opline)))) {
				Z_TYPE_FLAGS(ZEND_OP2_LITERAL(opline)) &= ~ (IS_TYPE_REFCOUNTED | IS_TYPE_COPYABLE);
			}
			ZVAL_NULL(&ZEND_OP2_LITERAL(src));
#else
			Z_STR(ZEND_OP2_LITERAL(opline)) = Z_STR(ZEND_OP2_LITERAL(src));
#endif
			MAKE_NOP(src);
		} else if ((opline->opcode == ZEND_ADD_STRING || opline->opcode == ZEND_ADD_VAR) && ZEND_OP1_TYPE(opline) == IS_CONST) {
			/* convert ADD_STRING(C1, C2) to CONCAT(C1, C2) */
			opline->opcode = ZEND_CONCAT;
			continue;
		} else if (opline->opcode == ZEND_ADD_CHAR && ZEND_OP1_TYPE(opline) == IS_CONST && ZEND_OP2_TYPE(opline) == IS_CONST) {
            /* convert ADD_CHAR(C1, C2) to CONCAT(C1, C2) */
			char c = (char)Z_LVAL(ZEND_OP2_LITERAL(opline));
			ZVAL_STRINGL(&ZEND_OP2_LITERAL(opline), &c, 1);
			opline->opcode = ZEND_CONCAT;
			continue;
		} else if ((opline->opcode == ZEND_ADD ||
					opline->opcode == ZEND_SUB ||
					opline->opcode == ZEND_MUL ||
					opline->opcode == ZEND_DIV ||
					opline->opcode == ZEND_MOD ||
					opline->opcode == ZEND_SL ||
					opline->opcode == ZEND_SR ||
					opline->opcode == ZEND_CONCAT ||
					opline->opcode == ZEND_IS_EQUAL ||
					opline->opcode == ZEND_IS_NOT_EQUAL ||
					opline->opcode == ZEND_IS_SMALLER ||
					opline->opcode == ZEND_IS_SMALLER_OR_EQUAL ||
					opline->opcode == ZEND_IS_IDENTICAL ||
					opline->opcode == ZEND_IS_NOT_IDENTICAL ||
					opline->opcode == ZEND_BOOL_XOR ||
					opline->opcode == ZEND_BW_OR ||
					opline->opcode == ZEND_BW_AND ||
					opline->opcode == ZEND_BW_XOR) &&
					ZEND_OP1_TYPE(opline)==IS_CONST &&
					ZEND_OP2_TYPE(opline)==IS_CONST) {
			/* evaluate constant expressions */
			int (*binary_op)(zval *result, zval *op1, zval *op2 TSRMLS_DC) = get_binary_op(opline->opcode);
			zval result;
			int er;

            if ((opline->opcode == ZEND_DIV || opline->opcode == ZEND_MOD) &&
                ((Z_TYPE(ZEND_OP2_LITERAL(opline)) == IS_LONG &&
                  Z_LVAL(ZEND_OP2_LITERAL(opline)) == 0) ||
                 (Z_TYPE(ZEND_OP2_LITERAL(opline)) == IS_DOUBLE &&
                  Z_DVAL(ZEND_OP2_LITERAL(opline)) == 0.0))) {
				if (RESULT_USED(opline)) {
					SET_VAR_SOURCE(opline);
				}
                opline++;
				continue;
			}
			er = EG(error_reporting);
			EG(error_reporting) = 0;
			if (binary_op(&result, &ZEND_OP1_LITERAL(opline), &ZEND_OP2_LITERAL(opline) TSRMLS_CC) == SUCCESS) {
//???				PZ_SET_REFCOUNT_P(&result, 1);
//???				PZ_UNSET_ISREF_P(&result);

				literal_dtor(&ZEND_OP1_LITERAL(opline));
				literal_dtor(&ZEND_OP2_LITERAL(opline));
				opline->opcode = ZEND_QM_ASSIGN;
				SET_UNUSED(opline->op2);
				update_op1_const(op_array, opline, &result TSRMLS_CC);
			}
			EG(error_reporting) = er;
		} else if ((opline->opcode == ZEND_BOOL ||
				   	opline->opcode == ZEND_BOOL_NOT ||
				  	opline->opcode == ZEND_BW_NOT) && ZEND_OP1_TYPE(opline) == IS_CONST) {
			/* evaluate constant unary ops */
			unary_op_type unary_op = get_unary_op(opline->opcode);
			zval result;

			if (unary_op) {
#if ZEND_EXTENSION_API_NO < PHP_5_3_X_API_NO
				unary_op(&result, &ZEND_OP1_LITERAL(opline));
#else
				unary_op(&result, &ZEND_OP1_LITERAL(opline) TSRMLS_CC);
#endif
				literal_dtor(&ZEND_OP1_LITERAL(opline));
			} else {
				/* BOOL */
				result = ZEND_OP1_LITERAL(opline);
				convert_to_boolean(&result);
				ZVAL_NULL(&ZEND_OP1_LITERAL(opline));
			}
//???			PZ_SET_REFCOUNT_P(&result, 1);
//???			PZ_UNSET_ISREF_P(&result);
			opline->opcode = ZEND_QM_ASSIGN;
			update_op1_const(op_array, opline, &result TSRMLS_CC);
		} else if ((opline->opcode == ZEND_RETURN || opline->opcode == ZEND_EXIT) &&
					ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
				   	VAR_SOURCE(opline->op1) &&
				   	VAR_SOURCE(opline->op1)->opcode == ZEND_QM_ASSIGN) {
			/* T = QM_ASSIGN(X), RETURN(T) to RETURN(X) */
			zend_op *src = VAR_SOURCE(opline->op1);
			VAR_UNSET(opline->op1);
			COPY_NODE(opline->op1, src->op1);
			MAKE_NOP(src);
		} else if ((opline->opcode == ZEND_ADD_STRING ||
					opline->opcode == ZEND_ADD_CHAR) &&
				  	ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
				  	VAR_SOURCE(opline->op1) &&
				  	VAR_SOURCE(opline->op1)->opcode == ZEND_INIT_STRING) {
			/* convert T = INIT_STRING(), T = ADD_STRING(T, X) to T = QM_ASSIGN(X) */
			/* CHECKME: Remove ZEND_ADD_VAR optimization, since some conversions -
			   namely, BOOL(false)->string - don't allocate memory but use empty_string
			   and ADD_CHAR fails */
			zend_op *src = VAR_SOURCE(opline->op1);
			VAR_UNSET(opline->op1);
			COPY_NODE(opline->op1, opline->op2);
			if (opline->opcode == ZEND_ADD_CHAR) {
				char c = (char)Z_LVAL(ZEND_OP2_LITERAL(opline));
				ZVAL_STRINGL(&ZEND_OP1_LITERAL(opline), &c, 1);
			}
			SET_UNUSED(opline->op2);
			MAKE_NOP(src);
			opline->opcode = ZEND_QM_ASSIGN;
		} else if ((opline->opcode == ZEND_ADD_STRING ||
				   	opline->opcode == ZEND_ADD_CHAR ||
				   	opline->opcode == ZEND_ADD_VAR ||
				   	opline->opcode == ZEND_CONCAT) &&
				  	ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
				  	VAR_SOURCE(opline->op1) &&
				  	VAR_SOURCE(opline->op1)->opcode == ZEND_CONCAT &&
				  	ZEND_OP2_TYPE(VAR_SOURCE(opline->op1)) == IS_CONST &&
				  	Z_TYPE(ZEND_OP2_LITERAL(VAR_SOURCE(opline->op1))) == IS_STRING &&
				  	Z_STRLEN(ZEND_OP2_LITERAL(VAR_SOURCE(opline->op1))) == 0) {
			/* convert T = CONCAT(X,''), T = ADD_STRING(T, Y) to T = CONCAT(X,Y) */
			zend_op *src = VAR_SOURCE(opline->op1);
			VAR_UNSET(opline->op1);
			COPY_NODE(opline->op1, src->op1);
			if (opline->opcode == ZEND_ADD_CHAR) {
				char c = (char)Z_LVAL(ZEND_OP2_LITERAL(opline));
				ZVAL_STRINGL(&ZEND_OP2_LITERAL(opline), &c, 1);
			}
			opline->opcode = ZEND_CONCAT;
			literal_dtor(&ZEND_OP2_LITERAL(src)); /* will take care of empty_string too */
			MAKE_NOP(src);
//??? This optimization can't work anymore because ADD_VAR returns IS_TMP_VAR
//??? and ZEND_CAST returns IS_VAR.
//??? BTW: it wan't used for long time, because we don't use INIT_STRING
#if 0
		} else if (opline->opcode == ZEND_ADD_VAR &&
					ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
					VAR_SOURCE(opline->op1) &&
					VAR_SOURCE(opline->op1)->opcode == ZEND_INIT_STRING) {
			/* convert T = INIT_STRING(), T = ADD_VAR(T, X) to T = CAST(STRING, X) */
			zend_op *src = VAR_SOURCE(opline->op1);
			VAR_UNSET(opline->op1);
			COPY_NODE(opline->op1, opline->op2);
			SET_UNUSED(opline->op2);
			MAKE_NOP(src);
			opline->opcode = ZEND_CAST;
			opline->extended_value = IS_STRING;
#endif
		} else if ((opline->opcode == ZEND_ADD_STRING ||
					opline->opcode == ZEND_ADD_CHAR ||
					opline->opcode == ZEND_ADD_VAR ||
					opline->opcode == ZEND_CONCAT) &&
					ZEND_OP1_TYPE(opline) == (IS_TMP_VAR|IS_VAR) &&
					VAR_SOURCE(opline->op1) &&
					VAR_SOURCE(opline->op1)->opcode == ZEND_CAST &&
					VAR_SOURCE(opline->op1)->extended_value == IS_STRING) {
			/* convert T1 = CAST(STRING, X), T2 = CONCAT(T1, Y) to T2 = CONCAT(X,Y) */
			zend_op *src = VAR_SOURCE(opline->op1);
			VAR_UNSET(opline->op1);
			COPY_NODE(opline->op1, src->op1);
			if (opline->opcode == ZEND_ADD_CHAR) {
				char c = (char)Z_LVAL(ZEND_OP2_LITERAL(opline));
				ZVAL_STRINGL(&ZEND_OP2_LITERAL(opline), &c, 1);
			}
			opline->opcode = ZEND_CONCAT;
			MAKE_NOP(src);
		} else if (opline->opcode == ZEND_QM_ASSIGN &&
					ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
					ZEND_RESULT_TYPE(opline) == IS_TMP_VAR &&
					ZEND_OP1(opline).var == ZEND_RESULT(opline).var) {
			/* strip T = QM_ASSIGN(T) */
			MAKE_NOP(opline);
		} else if (opline->opcode == ZEND_BOOL &&
					ZEND_OP1_TYPE(opline) == IS_TMP_VAR &&
					VAR_SOURCE(opline->op1) &&
					(VAR_SOURCE(opline->op1)->opcode == ZEND_IS_EQUAL ||
					VAR_SOURCE(opline->op1)->opcode == ZEND_IS_NOT_EQUAL ||
					VAR_SOURCE(opline->op1)->opcode == ZEND_IS_SMALLER ||
					VAR_SOURCE(opline->op1)->opcode == ZEND_IS_SMALLER_OR_EQUAL ||
					VAR_SOURCE(opline->op1)->opcode == ZEND_BOOL ||
					VAR_SOURCE(opline->op1)->opcode == ZEND_IS_IDENTICAL ||
					VAR_SOURCE(opline->op1)->opcode == ZEND_IS_NOT_IDENTICAL ||
					VAR_SOURCE(opline->op1)->opcode == ZEND_ISSET_ISEMPTY_VAR ||
					VAR_SOURCE(opline->op1)->opcode == ZEND_ISSET_ISEMPTY_DIM_OBJ) &&
					!used_ext[VAR_NUM(ZEND_OP1(opline).var)]) {
