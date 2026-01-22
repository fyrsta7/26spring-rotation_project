CDB::CDB(const char *pszFile, const char* pszMode) : pdb(NULL)
{
    int ret;
    if (pszFile == NULL)
        return;

    fReadOnly = (!strchr(pszMode, '+') && !strchr(pszMode, 'w'));
    bool fCreate = strchr(pszMode, 'c');
    unsigned int nFlags = DB_THREAD;
    if (fCreate)
        nFlags |= DB_CREATE;

    unsigned int nEnvFlags = 0;
    if (GetBoolArg("-privdb", true))
        nEnvFlags |= DB_PRIVATE;

    {
        LOCK(cs_db);
        if (!fDbEnvInit)
        {
            if (fShutdown)
                return;
            filesystem::path pathDataDir = GetDataDir();
            filesystem::path pathLogDir = pathDataDir / "database";
            filesystem::create_directory(pathLogDir);
            filesystem::path pathErrorFile = pathDataDir / "db.log";
            printf("dbenv.open LogDir=%s ErrorFile=%s\n", pathLogDir.string().c_str(), pathErrorFile.string().c_str());

            int nDbCache = GetArg("-dbcache", 25);
            dbenv.set_lg_dir(pathLogDir.string().c_str());
            dbenv.set_cachesize(nDbCache / 1024, (nDbCache % 1024)*1048576, 1);
            dbenv.set_lg_bsize(1048576);
            dbenv.set_lg_max(10485760);
            dbenv.set_lk_max_locks(10000);
            dbenv.set_lk_max_objects(10000);
            dbenv.set_errfile(fopen(pathErrorFile.string().c_str(), "a")); /// debug
            dbenv.set_flags(DB_TXN_WRITE_NOSYNC, 1);
            dbenv.set_flags(DB_AUTO_COMMIT, 1);
            dbenv.log_set_config(DB_LOG_AUTO_REMOVE, 1);
            ret = dbenv.open(pathDataDir.string().c_str(),
                             DB_CREATE     |
                             DB_INIT_LOCK  |
                             DB_INIT_LOG   |
                             DB_INIT_MPOOL |
                             DB_INIT_TXN   |
                             DB_THREAD     |
                             DB_RECOVER    |
                             nEnvFlags,
                             S_IRUSR | S_IWUSR);
            if (ret > 0)
                throw runtime_error(strprintf("CDB() : error %d opening database environment", ret));
            fDbEnvInit = true;
        }

        strFile = pszFile;
        ++mapFileUseCount[strFile];
        pdb = mapDb[strFile];
        if (pdb == NULL)
        {
            pdb = new Db(&dbenv, 0);

            ret = pdb->open(NULL,      // Txn pointer
                            pszFile,   // Filename
                            "main",    // Logical db name
                            DB_BTREE,  // Database type
                            nFlags,    // Flags
                            0);

            if (ret > 0)
            {
                delete pdb;
                pdb = NULL;
                {
                     LOCK(cs_db);
                    --mapFileUseCount[strFile];
                }
                strFile = "";
                throw runtime_error(strprintf("CDB() : can't open database file %s, error %d", pszFile, ret));
            }

            if (fCreate && !Exists(string("version")))
            {
                bool fTmp = fReadOnly;
                fReadOnly = false;
                WriteVersion(CLIENT_VERSION);
                fReadOnly = fTmp;
            }

            mapDb[strFile] = pdb;
        }
    }
}
