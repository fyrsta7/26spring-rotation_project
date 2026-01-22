
/*
 * Close the sqlite database
 */

void sql_close_database(void)
{
    int rc;
    if (unlikely(!db_meta))
        return;

    netdata_log_info("Closing SQLite database");

    add_stmt_to_list(NULL);
