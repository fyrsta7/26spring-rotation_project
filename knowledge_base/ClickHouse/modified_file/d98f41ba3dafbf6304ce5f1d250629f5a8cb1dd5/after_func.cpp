		limits.max_bytes_to_read = settings.limits.max_bytes_to_read;
		limits.read_overflow_mode = settings.limits.read_overflow_mode;
		limits.max_execution_time = settings.limits.max_execution_time;
		limits.timeout_overflow_mode = settings.limits.timeout_overflow_mode;
		limits.min_execution_speed = settings.limits.min_execution_speed;
		limits.timeout_before_checking_execution_speed = settings.limits.timeout_before_checking_execution_speed;

		QuotaForIntervals & quota = context.getQuota();
		
		for (BlockInputStreams::iterator it = streams.begin(); it != streams.end(); ++it)
		{
			if (IProfilingBlockInputStream * stream = dynamic_cast<IProfilingBlockInputStream *>(&**it))
			{
				stream->setLimits(limits);
				stream->setQuota(quota, IProfilingBlockInputStream::QUOTA_READ);
			}
		}
	}

	return from_stage;
}


void InterpreterSelectQuery::executeWhere(BlockInputStreams & streams, ExpressionActionsPtr expression)
{
	bool is_async = settings.asynchronous && streams.size() <= settings.max_threads;
	for (BlockInputStreams::iterator it = streams.begin(); it != streams.end(); ++it)
	{
		BlockInputStreamPtr & stream = *it;
		stream = maybeAsynchronous(new ExpressionBlockInputStream(stream, expression), is_async);
		stream = maybeAsynchronous(new FilterBlockInputStream(stream, query.where_expression->getColumnName()), is_async);
	}
}


void InterpreterSelectQuery::executeAggregation(BlockInputStreams & streams, ExpressionActionsPtr expression)
{
	bool is_async = settings.asynchronous && streams.size() <= settings.max_threads;
	for (BlockInputStreams::iterator it = streams.begin(); it != streams.end(); ++it)
	{
		BlockInputStreamPtr & stream = *it;
		stream = maybeAsynchronous(new ExpressionBlockInputStream(stream, expression), is_async);
