        arg_item->get_settable_routine_parameter();

      DBUG_ASSERT(srp);

      if (srp->set_value(thd, octx, nctx->get_item_addr(i)))
      {
        err_status= TRUE;
        break;
      }

      Send_field *out_param_info= new (thd->mem_root) Send_field();
      nctx->get_item(i)->make_field(out_param_info);
      out_param_info->db_name= m_db.str;
      out_param_info->table_name= m_name.str;
      out_param_info->org_table_name= m_name.str;
      out_param_info->col_name= spvar->name.str;
      out_param_info->org_col_name= spvar->name.str;

      srp->set_out_param_info(out_param_info);
    }
  }

#ifndef NO_EMBEDDED_ACCESS_CHECKS
  if (save_security_ctx)
    m_security_ctx.restore_security_context(thd, save_security_ctx);
#endif

  if (!save_spcont)
    delete octx;

  delete nctx;
  thd->spcont= save_spcont;
  thd->utime_after_lock= utime_before_sp_exec;

  /*
    If not insided a procedure and a function printing warning
    messsages.
  */ 
  bool need_binlog_call= mysql_bin_log.is_open() &&
                         (thd->variables.option_bits & OPTION_BIN_LOG) &&
                         !thd->is_current_stmt_binlog_format_row();
  if (need_binlog_call && thd->spcont == NULL &&
      !thd->binlog_evt_union.do_union)
    thd->issue_unsafe_warnings();

  DBUG_RETURN(err_status);
}


/**
  Reset lex during parsing, before we parse a sub statement.

  @param thd Thread handler.

  @return Error state
    @retval true An error occurred.
    @retval false Success.
*/

bool
sp_head::reset_lex(THD *thd)
{
  DBUG_ENTER("sp_head::reset_lex");
  LEX *sublex;
  LEX *oldlex= thd->lex;

  sublex= new (thd->mem_root)st_lex_local;
  if (sublex == 0)
    DBUG_RETURN(TRUE);

  thd->lex= sublex;
  (void)m_lex.push_front(oldlex);

  /* Reset most stuff. */
  lex_start(thd);

  /* And keep the SP stuff too */
  sublex->sphead= oldlex->sphead;
  sublex->spcont= oldlex->spcont;
  /* And trigger related stuff too */
  sublex->trg_chistics= oldlex->trg_chistics;
  sublex->trg_table_fields.empty();
  sublex->sp_lex_in_use= FALSE;

  /* Reset type info. */

  sublex->charset= NULL;
  sublex->length= NULL;
  sublex->dec= NULL;
  sublex->interval_list.empty();
  sublex->type= 0;

  /* Reset part of parser state which needs this. */
  thd->m_parser_state->m_yacc.reset_before_substatement();

  DBUG_RETURN(FALSE);
}


/**
