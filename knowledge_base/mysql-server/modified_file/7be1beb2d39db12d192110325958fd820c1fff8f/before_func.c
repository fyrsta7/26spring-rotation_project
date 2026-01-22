	char*	name)	/* in: table name */
{
	ulint	error;
	trx_t*	trx;

	trx = trx_allocate_for_background();

/*	fprintf(stderr, "InnoDB: Dropping table %s in background drop list\n",
							name); */
  	/* Drop the table in InnoDB */

  	error = row_drop_table_for_mysql(name, trx);

	if (error != DB_SUCCESS) {
		fprintf(stderr,
	"InnoDB: Error: Dropping table %s in background drop list failed\n",
								name);
	}
  	
	/* Flush the log to reduce probability that the .frm files and
	the InnoDB data dictionary get out-of-sync if the user runs
	with innodb_flush_log_at_trx_commit = 0 */
	
	log_buffer_flush_to_disk();

  	trx_commit_for_mysql(trx);

  	trx_free_for_background(trx);

	return(DB_SUCCESS);
}

/*************************************************************************
The master thread in srv0srv.c calls this regularly to drop tables which
we must drop in background after queries to them have ended. Such lazy
dropping of tables is needed in ALTER TABLE on Unix. */

ulint
row_drop_tables_for_mysql_in_background(void)
/*=========================================*/
					/* out: how many tables dropped
					+ remaining tables in list */
{
	row_mysql_drop_t*	drop;
	dict_table_t*		table;
	ulint			n_tables;
	ulint			n_tables_dropped = 0;
loop:	
	mutex_enter(&kernel_mutex);

	if (!row_mysql_drop_list_inited) {

		UT_LIST_INIT(row_mysql_drop_list);
		row_mysql_drop_list_inited = TRUE;
	}

	drop = UT_LIST_GET_FIRST(row_mysql_drop_list);
	
	n_tables = UT_LIST_GET_LEN(row_mysql_drop_list);

	mutex_exit(&kernel_mutex);

	if (drop == NULL) {

		return(n_tables + n_tables_dropped);
	}

	mutex_enter(&(dict_sys->mutex));
	table = dict_table_get_low(drop->table_name);
	mutex_exit(&(dict_sys->mutex));

	if (table == NULL) {
	        /* If for some reason the table has already been dropped
		through some other mechanism, do not try to drop it */

	        goto already_dropped;
	}

	if (table->n_mysql_handles_opened > 0
				|| table->n_foreign_key_checks_running > 0) {

		return(n_tables + n_tables_dropped);
	}

	n_tables_dropped++;
							
	row_drop_table_for_mysql_in_background(drop->table_name);

already_dropped:
	mutex_enter(&kernel_mutex);

	UT_LIST_REMOVE(row_mysql_drop_list, row_mysql_drop_list, drop);

        ut_print_timestamp(stderr);
        fprintf(stderr,
		"  InnoDB: Dropped table %s in background drop queue.\n",
		drop->table_name);

	mem_free(drop->table_name);

	mem_free(drop);

	mutex_exit(&kernel_mutex);

	goto loop;
}

/*************************************************************************
Get the background drop list length. NOTE: the caller must own the kernel
mutex! */

ulint
row_get_background_drop_list_len_low(void)
/*======================================*/
					/* out: how many tables in list */
{
	ut_ad(mutex_own(&kernel_mutex));

	if (!row_mysql_drop_list_inited) {

		UT_LIST_INIT(row_mysql_drop_list);
		row_mysql_drop_list_inited = TRUE;
	}
	
	return(UT_LIST_GET_LEN(row_mysql_drop_list));
}

/*************************************************************************
Adds a table to the list of tables which the master thread drops in
background. We need this on Unix because in ALTER TABLE MySQL may call
drop table even if the table has running queries on it. */
static
void
row_add_table_to_background_drop_list(
/*==================================*/
	dict_table_t*	table)	/* in: table */
{
	row_mysql_drop_t*	drop;
	
	drop = mem_alloc(sizeof(row_mysql_drop_t));

	drop->table_name = mem_alloc(1 + ut_strlen(table->name));

	ut_memcpy(drop->table_name, table->name, 1 + ut_strlen(table->name));

	mutex_enter(&kernel_mutex);

	if (!row_mysql_drop_list_inited) {

		UT_LIST_INIT(row_mysql_drop_list);
		row_mysql_drop_list_inited = TRUE;
	}

	UT_LIST_ADD_LAST(row_mysql_drop_list, row_mysql_drop_list, drop);
	
/*	fprintf(stderr, "InnoDB: Adding table %s to background drop list\n",
							drop->table_name); */
	mutex_exit(&kernel_mutex);
}

/*************************************************************************
Drops a table for MySQL. If the name of the dropped table ends to
characters INNODB_MONITOR, then this also stops printing of monitor
output by the master thread. */

int
row_drop_table_for_mysql(
/*=====================*/
				/* out: error code or DB_SUCCESS */
	char*	name,		/* in: table name */
	trx_t*	trx)		/* in: transaction handle */
{
	dict_table_t*	table;
	que_thr_t*	thr;
	que_t*		graph;
	ulint		err;
	char*		str1;
	char*		str2;
	ulint		len;
	ulint		namelen;
	ulint		keywordlen;
	ibool		locked_dictionary	= FALSE;
	char		buf[10000];

	ut_ad(trx->mysql_thread_id == os_thread_get_curr_id());
	ut_a(name != NULL);

	if (srv_created_new_raw) {
		fprintf(stderr,
		"InnoDB: A new raw disk partition was initialized or\n"
		"InnoDB: innodb_force_recovery is on: we do not allow\n"
		"InnoDB: database modifications by the user. Shut down\n"
		"InnoDB: mysqld and edit my.cnf so that newraw is replaced\n"
		"InnoDB: with raw, and innodb_force_... is removed.\n");

		return(DB_ERROR);
	}

	trx->op_info = (char *) "dropping table";

	trx_start_if_not_started(trx);

	namelen = ut_strlen(name);
	keywordlen = ut_strlen((char *) "innodb_monitor");

	if (namelen >= keywordlen
	    && 0 == ut_memcmp(name + namelen - keywordlen,
			      (char *) "innodb_monitor", keywordlen)) {

		/* Table name ends to characters innodb_monitor:
		stop monitor prints */
 				
		srv_print_innodb_monitor = FALSE;
		srv_print_innodb_lock_monitor = FALSE;
	}

	keywordlen = ut_strlen((char *) "innodb_lock_monitor");

	if (namelen >= keywordlen
		    && 0 == ut_memcmp(name + namelen - keywordlen,
				      (char *) "innodb_lock_monitor",
				      keywordlen)) {

		srv_print_innodb_monitor = FALSE;
		srv_print_innodb_lock_monitor = FALSE;
	}

	keywordlen = ut_strlen((char *) "innodb_tablespace_monitor");

	if (namelen >= keywordlen
		    && 0 == ut_memcmp(name + namelen - keywordlen,
				      (char *) "innodb_tablespace_monitor",
				      keywordlen)) {

		srv_print_innodb_tablespace_monitor = FALSE;
	}

	keywordlen = ut_strlen((char *) "innodb_table_monitor");

	if (namelen >= keywordlen
		    && 0 == ut_memcmp(name + namelen - keywordlen,
				      (char *) "innodb_table_monitor",
				      keywordlen)) {

		srv_print_innodb_table_monitor = FALSE;
	}

	/* We use the private SQL parser of Innobase to generate the
	query graphs needed in deleting the dictionary data from system
	tables in Innobase. Deleting a row from SYS_INDEXES table also
	frees the file segments of the B-tree associated with the index. */

	str1 = (char *) 
	"PROCEDURE DROP_TABLE_PROC () IS\n"
	"table_name CHAR;\n"
	"sys_foreign_id CHAR;\n"
	"table_id CHAR;\n"
	"index_id CHAR;\n"
	"foreign_id CHAR;\n"
	"found INT;\n"
	"BEGIN\n"
	"table_name := '";
	
	str2 = (char *) 
	"';\n"
	"SELECT ID INTO table_id\n"
	"FROM SYS_TABLES\n"
	"WHERE NAME = table_name;\n"
	"IF (SQL % NOTFOUND) THEN\n"
	"	COMMIT WORK;\n"
	"	RETURN;\n"
