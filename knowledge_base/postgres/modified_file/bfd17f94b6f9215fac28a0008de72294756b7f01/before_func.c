		appendPQExpBuffer(creaQry, " TABLESPACE = %s",
						  fmtId(tablespace));
	appendPQExpBuffer(creaQry, ";\n");

	if (binary_upgrade)
	{
		appendPQExpBuffer(creaQry, "\n-- For binary upgrade, set datfrozenxid.\n");
		appendPQExpBuffer(creaQry, "UPDATE pg_database\n"
							 "SET datfrozenxid = '%u'\n"
							 "WHERE	datname = '%s';\n",
							 frozenxid, datname);
	}
	
	appendPQExpBuffer(delQry, "DROP DATABASE %s;\n",
					  fmtId(datname));

	dbDumpId = createDumpId();

	ArchiveEntry(AH,
				 dbCatId,		/* catalog ID */
				 dbDumpId,		/* dump ID */
				 datname,		/* Name */
				 NULL,			/* Namespace */
				 NULL,			/* Tablespace */
				 dba,			/* Owner */
				 false,			/* with oids */
				 "DATABASE",	/* Desc */
				 SECTION_PRE_DATA, /* Section */
				 creaQry->data, /* Create */
				 delQry->data,	/* Del */
				 NULL,			/* Copy */
				 NULL,			/* Deps */
				 0,				/* # Deps */
				 NULL,			/* Dumper */
				 NULL);			/* Dumper Arg */

	/* Dump DB comment if any */
	if (g_fout->remoteVersion >= 80200)
	{
		/*
		 * 8.2 keeps comments on shared objects in a shared table, so we
		 * cannot use the dumpComment used for other database objects.
		 */
		char	   *comment = PQgetvalue(res, 0, PQfnumber(res, "description"));

		if (comment && strlen(comment))
		{
			resetPQExpBuffer(dbQry);
			/* Generates warning when loaded into a differently-named database.*/
			appendPQExpBuffer(dbQry, "COMMENT ON DATABASE %s IS ", fmtId(datname));
			appendStringLiteralAH(dbQry, comment, AH);
			appendPQExpBuffer(dbQry, ";\n");

			ArchiveEntry(AH, dbCatId, createDumpId(), datname, NULL, NULL,
						 dba, false, "COMMENT", SECTION_NONE,
						 dbQry->data, "", NULL,
						 &dbDumpId, 1, NULL, NULL);
		}
	}
	else
	{
		resetPQExpBuffer(dbQry);
		appendPQExpBuffer(dbQry, "DATABASE %s", fmtId(datname));
		dumpComment(AH, dbQry->data, NULL, "",
					dbCatId, 0, dbDumpId);
	}

	PQclear(res);

	destroyPQExpBuffer(dbQry);
	destroyPQExpBuffer(delQry);
	destroyPQExpBuffer(creaQry);
}


/*
 * dumpEncoding: put the correct encoding into the archive
