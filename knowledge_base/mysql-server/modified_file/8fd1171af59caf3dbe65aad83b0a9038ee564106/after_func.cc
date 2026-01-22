  {
    my_error(ER_OPTION_PREVENTS_STATEMENT, MYF(0),
             "--skip-grant-tables");
    DBUG_RETURN(TRUE);
  }

  if (rights & ~PROC_ACLS)
  {
    my_error(ER_ILLEGAL_GRANT_FOR_TABLE, MYF(0));
    DBUG_RETURN(TRUE);
  }

  if (!revoke_grant)
  {
    if (sp_exist_routines(thd, table_list, is_proc))
      DBUG_RETURN(TRUE);
  }

  /*
    This statement will be replicated as a statement, even when using
    row-based replication.  The binlog state will be cleared here to
    statement based replication and will be reset to the originals
    values when we are out of this function scope
  */
  Save_and_Restore_binlog_format_state binlog_format_state(thd);
  if ((ret= open_grant_tables(thd, tables, &transactional_tables)))
    DBUG_RETURN(ret != 1);

  Acl_cache_lock_guard acl_cache_lock(thd, Acl_cache_lock_mode::WRITE_MODE);
  if (!acl_cache_lock.lock())
  {
    commit_and_close_mysql_tables(thd);
    DBUG_RETURN(true);
  }

  if (!revoke_grant)
    create_new_users= test_if_create_new_users(thd);

  is_privileged_user= is_privileged_user_for_credential_change(thd);
  MEM_ROOT *old_root= thd->mem_root;
  thd->mem_root= &memex;

  DBUG_PRINT("info",("now time to iterate and add users"));

  while ((tmp_Str= str_list++))
  {
    int error;
    GRANT_NAME *grant_name;

    if (!(Str= get_current_user(thd, tmp_Str)))
    {
      result= true;
      continue;
    }

    if (set_and_validate_user_attributes(thd, Str, what_to_set,
                                         is_privileged_user, false,
                                         &tables[ACL_TABLES::TABLE_PASSWORD_HISTORY], NULL))
    {
      result= true;
      continue;
    }

    ACL_USER *this_user= find_acl_user(Str->host.str, Str->user.str, true);
    if (this_user && (what_to_set & PLUGIN_ATTR))
      existing_users.insert(tmp_Str);

    /* Create user if needed */
    if ((error= replace_user_table(thd, tables[ACL_TABLES::TABLE_USER].table, Str,
                                   0, revoke_grant, create_new_users,
                                   what_to_set)))
    {
      result= true;                             // Remember error
      if (error < 0)
        break;

      continue;
    }
    db_name= table_list->db;
    if (write_to_binlog)
      thd->add_to_binlog_accessed_dbs(db_name);
    table_name= table_list->table_name;
    grant_name= routine_hash_search(Str->host.str, NullS, db_name,
                                    Str->user.str, table_name, is_proc, 1);
    if (!grant_name)
    {
      if (revoke_grant)
      {
        my_error(ER_NONEXISTING_PROC_GRANT, MYF(0),
                 Str->user.str, Str->host.str, table_name);
        result= true;
        continue;
      }
      grant_name= new (*THR_MALLOC) GRANT_NAME(Str->host.str, db_name,
                                               Str->user.str, table_name,
                                               rights, TRUE);
      if (!grant_name)
      {
        result= true;
        break;
      }
      if (is_proc)
        proc_priv_hash->emplace(grant_name->hash_key,
                                unique_ptr_destroy_only<GRANT_NAME>(grant_name));
      else
        func_priv_hash->emplace(grant_name->hash_key,
                                unique_ptr_destroy_only<GRANT_NAME>(grant_name));
    }

    if ((error= replace_routine_table(thd, grant_name, tables[4].table, *Str,
                                      db_name, table_name, is_proc, rights,
                                      revoke_grant)))
    {
      result= true;                             // Remember error
      if (error < 0)
        break;

      continue;
    }
  }
  thd->mem_root= old_root;

  /*
    mysql_routine_grant can be called in following scenarios:
    1. As a part of GRANT statement
    2. As a part of CREATE PROCEDURE/ROUTINE statement

    In case of 2, even if we fail to grant permission on
    newly created routine, it is not a critical error and
    is suppressed by caller. Instead, a warning is thrown
    to user.

    So, if we are here and result is set to true, either of the following must be true:
    1. An error is set in THD
    2. Current statement is SQLCOM_CREATE_PROCEDURE or SQLCOM_CREATE_SPFUNCTION

    So assert for the same.
  */
  DBUG_ASSERT(!result || thd->is_error() ||
              thd->lex->sql_command == SQLCOM_CREATE_PROCEDURE ||
              thd->lex->sql_command == SQLCOM_CREATE_SPFUNCTION);

  result= log_and_commit_acl_ddl(thd, transactional_tables, NULL, result,
                                 write_to_binlog, write_to_binlog);

  {
    /* Notify audit plugin. We will ignore the return value. */
    for (LEX_USER * one_user : existing_users)
    {
      LEX_USER * existing_user;
      if ((existing_user= get_current_user(thd, one_user)))
        mysql_audit_notify(thd, AUDIT_EVENT(MYSQL_AUDIT_AUTHENTICATION_CREDENTIAL_CHANGE),
                           thd->is_error(),
                           existing_user->user.str,
                           existing_user->host.str,
                           existing_user->plugin.str,
