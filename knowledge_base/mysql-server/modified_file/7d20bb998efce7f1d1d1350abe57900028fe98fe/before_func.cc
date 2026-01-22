*/

void mysql_sql_stmt_prepare(THD *thd)
{
  LEX *lex= thd->lex;
  LEX_STRING *name= &lex->prepared_stmt_name;
  Prepared_statement *stmt;
  const char *query;
  uint query_len= 0;
  DBUG_ENTER("mysql_sql_stmt_prepare");

  if ((stmt= (Prepared_statement*) thd->stmt_map.find_by_name(name)))
  {
    /*
      If there is a statement with the same name, remove it. It is ok to
      remove old and fail to insert a new one at the same time.
    */
    if (stmt->is_in_use())
    {
      my_error(ER_PS_NO_RECURSION, MYF(0));
      DBUG_VOID_RETURN;
    }

    stmt->deallocate();
  }

  if (! (query= get_dynamic_sql_string(lex, &query_len)) ||
      ! (stmt= new Prepared_statement(thd)))
