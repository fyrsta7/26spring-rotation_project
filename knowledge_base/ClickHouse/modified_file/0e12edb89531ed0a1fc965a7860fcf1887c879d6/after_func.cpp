QueryPipeline InterpreterSelectWithUnionQuery::executeWithProcessors()
{
    QueryPipeline main_pipeline;
    std::vector<QueryPipeline> pipelines;
    bool has_main_pipeline = false;

    Blocks headers;
    headers.reserve(nested_interpreters.size());

    for (auto & interpreter : nested_interpreters)
    {
        if (!has_main_pipeline)
        {
            has_main_pipeline = true;
            main_pipeline = interpreter->executeWithProcessors();
            headers.emplace_back(main_pipeline.getHeader());
        }
        else
        {
            pipelines.emplace_back(interpreter->executeWithProcessors());
            headers.emplace_back(pipelines.back().getHeader());
        }
    }

    if (!has_main_pipeline)
        main_pipeline.init(Pipe(std::make_shared<NullSource>(getSampleBlock())));

    if (!pipelines.empty())
    {
        auto common_header = getCommonHeaderForUnion(headers);
        main_pipeline.unitePipelines(std::move(pipelines), common_header);

        // nested queries can force 1 thread (due to simplicity)
        // but in case of union this cannot be done.
        UInt64 max_threads = context->getSettingsRef().max_threads;
        main_pipeline.setMaxThreads(std::min(nested_interpreters.size(), max_threads));
    }

    main_pipeline.addInterpreterContext(context);

    return main_pipeline;
}
