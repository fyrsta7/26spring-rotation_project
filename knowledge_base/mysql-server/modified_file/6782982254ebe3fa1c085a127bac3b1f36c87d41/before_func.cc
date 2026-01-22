      while (pos > buf && !my_isgraph(&my_charset_latin1, pos[-1]))
	pos--;
      *pos=0;
      if ((pos= strchr(buf, '=')))
      {
	if (!strncmp(buf,"default-character-set", (pos-buf)))
	{
	  if (!(create->default_table_charset=
		get_charset_by_csname(pos+1, 
				      MY_CS_PRIMARY,
				      MYF(0))))
	  {
	    sql_print_error("Error while loading database options: '%s':",path);
	    sql_print_error(ER(ER_UNKNOWN_CHARACTER_SET),pos+1);
	  }
	}
	else if (!strncmp(buf,"default-collation", (pos-buf)))
	{
	  if (!(create->default_table_charset= get_charset_by_name(pos+1,
								   MYF(0))))
	  {
	    sql_print_error("Error while loading database options: '%s':",path);
	    sql_print_error(ER(ER_UNKNOWN_COLLATION),pos+1);
	  }
	}
      }
    }
    error=0;
    end_io_cache(&cache);
    my_close(file,MYF(0));
  }
  DBUG_RETURN(error);
}


/*
  Create a database

  SYNOPSIS
  mysql_create_db()
  thd		Thread handler
  db		Name of database to create
		Function assumes that this is already validated.
  create_info	Database create options (like character set)
  silent	Used by replication when internally creating a database.
		In this case the entry should not be logged.

  RETURN VALUES
  0	ok
  -1	Error

*/

int mysql_create_db(THD *thd, char *db, HA_CREATE_INFO *create_info,
		    bool silent)
{
  char	 path[FN_REFLEN+16];
  long result=1;
  int error = 0;
  MY_STAT stat_info;
  uint create_options = create_info ? create_info->options : 0;
  uint path_len;
  DBUG_ENTER("mysql_create_db");
  
  VOID(pthread_mutex_lock(&LOCK_mysql_create_db));

  // do not create database if another thread is holding read lock
  if (wait_if_global_read_lock(thd,0))
  {
    error= -1;
    goto exit2;
  }

  /* Check directory */
  strxmov(path, mysql_data_home, "/", db, NullS);
  path_len= unpack_dirname(path,path);    // Convert if not unix
  path[path_len-1]= 0;                    // Remove last '/' from path

  if (my_stat(path,&stat_info,MYF(0)))
  {
   if (!(create_options & HA_LEX_CREATE_IF_NOT_EXISTS))
    {
      my_error(ER_DB_CREATE_EXISTS,MYF(0),db);
      error = -1;
      goto exit;
    }
    result = 0;
  }
  else
  {
    if (my_errno != ENOENT)
    {
      my_error(EE_STAT, MYF(0),path,my_errno);
      goto exit;
