BlockInputStreamPtr InterpreterExplainQuery::executeImpl()
{
    const auto & ast = query->as<ASTExplainQuery &>();

    Block sample_block = getSampleBlock();
    MutableColumns res_columns = sample_block.cloneEmptyColumns();

    std::stringstream ss;

    if (ast.getKind() == ASTExplainQuery::ParsedAST)
    {
        if (ast.getSettings())
            throw Exception("Settings are not supported for EXPLAIN AST query.", ErrorCodes::UNKNOWN_SETTING);

        dumpAST(*ast.getExplainedQuery(), ss);
    }
    else if (ast.getKind() == ASTExplainQuery::AnalyzedSyntax)
    {
        if (ast.getSettings())
            throw Exception("Settings are not supported for EXPLAIN SYNTAX query.", ErrorCodes::UNKNOWN_SETTING);

        ExplainAnalyzedSyntaxVisitor::Data data{.context = context};
        ExplainAnalyzedSyntaxVisitor(data).visit(query);

        ast.getExplainedQuery()->format(IAST::FormatSettings(ss, false));
    }
    else if (ast.getKind() == ASTExplainQuery::QueryPlan)
    {
        if (!dynamic_cast<const ASTSelectWithUnionQuery *>(ast.getExplainedQuery().get()))
            throw Exception("Only SELECT is supported for EXPLAIN query", ErrorCodes::INCORRECT_QUERY);

        auto settings = checkAndGetSettings<QueryPlanSettings>(ast.getSettings());
        QueryPlan plan;

        InterpreterSelectWithUnionQuery interpreter(ast.getExplainedQuery(), context, SelectQueryOptions());
        interpreter.buildQueryPlan(plan);

        plan.optimize();

        WriteBufferFromOStream buffer(ss);
        plan.explainPlan(buffer, settings.query_plan_options);
    }
    else if (ast.getKind() == ASTExplainQuery::QueryPipeline)
    {
        if (!dynamic_cast<const ASTSelectWithUnionQuery *>(ast.getExplainedQuery().get()))
            throw Exception("Only SELECT is supported for EXPLAIN query", ErrorCodes::INCORRECT_QUERY);

        auto settings = checkAndGetSettings<QueryPipelineSettings>(ast.getSettings());
        QueryPlan plan;

        InterpreterSelectWithUnionQuery interpreter(ast.getExplainedQuery(), context, SelectQueryOptions());
        interpreter.buildQueryPlan(plan);
        auto pipeline = plan.buildQueryPipeline();

        WriteBufferFromOStream buffer(ss);

        if (settings.graph)
        {
            if (settings.compact)
                printPipelineCompact(pipeline->getProcessors(), buffer, settings.query_pipeline_options.header);
            else
                printPipeline(pipeline->getProcessors(), buffer);
        }
        else
        {
            plan.explainPipeline(buffer, settings.query_pipeline_options);
        }
    }

    fillColumn(*res_columns[0], ss.str());

    return std::make_shared<OneBlockInputStream>(sample_block.cloneWithColumns(std::move(res_columns)));
}
