}

static void datafile_delete(struct rrdengine_instance *ctx, struct rrdengine_datafile *datafile, bool worker) {
    if(worker)
        worker_is_busy(UV_EVENT_DATAFILE_ACQUIRE);

    bool datafile_got_for_deletion = datafile_acquire_for_deletion(datafile);

    if (ctx_is_available_for_queries(ctx))
        update_metrics_first_time_s(ctx, datafile, datafile->next, worker);

    while (!datafile_got_for_deletion) {
        if(worker)
            worker_is_busy(UV_EVENT_DATAFILE_ACQUIRE);

        datafile_got_for_deletion = datafile_acquire_for_deletion(datafile);

        if (!datafile_got_for_deletion) {
            info("DBENGINE: waiting for data file '%s/"
                         DATAFILE_PREFIX RRDENG_FILE_NUMBER_PRINT_TMPL DATAFILE_EXTENSION
                         "' to be available for deletion, "
                         "it is in use currently by %u users.",
                 ctx->dbfiles_path, ctx->datafiles.first->tier, ctx->datafiles.first->fileno, datafile->users.lockers);

            __atomic_add_fetch(&rrdeng_cache_efficiency_stats.datafile_deletion_spin, 1, __ATOMIC_RELAXED);
            sleep_usec(1 * USEC_PER_SEC);
        }
    }

    __atomic_add_fetch(&rrdeng_cache_efficiency_stats.datafile_deletion_started, 1, __ATOMIC_RELAXED);
    info("DBENGINE: deleting data file '%s/"
         DATAFILE_PREFIX RRDENG_FILE_NUMBER_PRINT_TMPL DATAFILE_EXTENSION
         "'.",
         ctx->dbfiles_path, ctx->datafiles.first->tier, ctx->datafiles.first->fileno);

    if(worker)
        worker_is_busy(UV_EVENT_DATAFILE_DELETE);

    struct rrdengine_journalfile *journal_file;
    unsigned deleted_bytes, journal_file_bytes, datafile_bytes;
    int ret;
    char path[RRDENG_PATH_MAX];

    uv_rwlock_wrlock(&ctx->datafiles.rwlock);

    journal_file = datafile->journalfile;
    datafile_bytes = datafile->pos;
    journal_file_bytes = journal_file->pos;
    deleted_bytes = journalfile_v2_data_size_get(journal_file);

    info("DBENGINE: deleting data and journal files to maintain disk quota");
    datafile_list_delete_unsafe(ctx, datafile);
    ret = journalfile_destroy_unsafe(journal_file, datafile);
    if (!ret) {
        journalfile_generate_path(datafile, path, sizeof(path));
        info("DBENGINE: deleted journal file \"%s\".", path);
        journalfile_v2_generate_path(datafile, path, sizeof(path));
        info("DBENGINE: deleted journal file \"%s\".", path);
        deleted_bytes += journal_file_bytes;
    }
    ret = destroy_data_file_unsafe(datafile);
    if (!ret) {
        generate_datafilepath(datafile, path, sizeof(path));
        info("DBENGINE: deleted data file \"%s\".", path);
        deleted_bytes += datafile_bytes;
    }
    freez(journal_file);
    freez(datafile);

    ctx->disk_space -= deleted_bytes;
    info("DBENGINE: reclaimed %u bytes of disk space.", deleted_bytes);
    uv_rwlock_wrunlock(&ctx->datafiles.rwlock);

