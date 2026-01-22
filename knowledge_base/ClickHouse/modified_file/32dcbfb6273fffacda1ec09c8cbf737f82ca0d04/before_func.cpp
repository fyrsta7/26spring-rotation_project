void DatabaseOrdinary::loadStoredObjects(
    Context & context,
    bool has_force_restore_data_flag)
{

    /** Tables load faster if they are loaded in sorted (by name) order.
      * Otherwise (for the ext4 filesystem), `DirectoryIterator` iterates through them in some order,
      *  which does not correspond to order tables creation and does not correspond to order of their location on disk.
      */
    using FileNames = std::map<std::string, ASTPtr>;
    FileNames file_names;

    size_t total_dictionaries = 0;
    iterateMetadataFiles(context, [context, &file_names, &total_dictionaries, this](const String & file_name)
    {
        String full_path = getMetadataPath() + file_name;
        try
        {
            auto ast = parseQueryFromMetadata(context, full_path, /*throw_on_error*/ true, /*remove_empty*/false);
            if (ast)
            {
                auto * create_query = ast->as<ASTCreateQuery>();
                file_names[file_name] = ast;
                total_dictionaries += create_query->is_dictionary;
            }
        }
        catch (Exception & e)
        {
            e.addMessage("Cannot parse definition from metadata file " + full_path);
            throw;
        }

    });

    size_t total_tables = file_names.size() - total_dictionaries;

    LOG_INFO(log, "Total " << total_tables << " tables and " << total_dictionaries << " dictionaries.");

    AtomicStopwatch watch;
    std::atomic<size_t> tables_processed{0};
    std::atomic<size_t> dictionaries_processed{0};

    ThreadPool pool(SettingMaxThreads().getAutoValue());

    /// Attach tables.
    for (const auto & name_with_query : file_names)
    {
        const auto & create_query = name_with_query.second->as<const ASTCreateQuery &>();
        if (!create_query.is_dictionary)
            pool.scheduleOrThrowOnError([&]()
            {
                tryAttachTable(context, create_query, *this, getDatabaseName(), has_force_restore_data_flag);

                /// Messages, so that it's not boring to wait for the server to load for a long time.
                logAboutProgress(log, ++tables_processed, total_tables, watch);
            });
    }

    pool.wait();

    /// After all tables was basically initialized, startup them.
    startupTables(pool);

    /// Attach dictionaries.
    attachToExternalDictionariesLoader(context);
    for (const auto & name_with_query : file_names)
    {
        auto create_query = name_with_query.second->as<const ASTCreateQuery &>();
        if (create_query.is_dictionary)
        {
            tryAttachDictionary(context, create_query, *this);

            /// Messages, so that it's not boring to wait for the server to load for a long time.
            logAboutProgress(log, ++dictionaries_processed, total_dictionaries, watch);
