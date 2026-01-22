void BackgroundProcessingPool::threadFunction()
{
	setThreadName("BackgrProcPool");

	std::mt19937 rng(reinterpret_cast<intptr_t>(&rng));
	std::this_thread::sleep_for(std::chrono::duration<double>(std::uniform_real_distribution<double>(0, sleep_seconds_random_part)(rng)));

	while (!shutdown)
	{
		Counters counters_diff;
		bool has_exception = false;

		try
		{
			TaskHandle task;
			time_t min_time = std::numeric_limits<time_t>::max();

			{
				std::unique_lock<std::mutex> lock(tasks_mutex);

				if (!tasks.empty())
				{
					/** Number of tasks is about number of tables of MergeTree family.
					  * Select task with minimal 'next_time_to_execute', and place to end of queue.
					  * Remind that one task could be selected and executed simultaneously from many threads.
					  *
					  * Tasks is like priority queue,
					  *  but we must have ability to change priority of any task in queue.
					  *
					  * If there is too much tasks, select from first 100.
					  * TODO Change list to multimap.
					  */
					size_t i = 0;
					for (const auto & handle : tasks)
					{
						if (handle->removed)
							continue;

						time_t next_time_to_execute = handle->next_time_to_execute;

						if (next_time_to_execute < min_time)
						{
							min_time = next_time_to_execute;
							task = handle;
						}

						++i;
						if (i > 100)
							break;
					}

					if (task)	/// Переложим в конец очереди (уменьшим приоритет среди задач с одинаковым next_time_to_execute).
						tasks.splice(tasks.end(), tasks, task->iterator);
				}
			}

			if (shutdown)
				break;

			if (!task)
			{
				std::unique_lock<std::mutex> lock(tasks_mutex);
				wake_event.wait_for(lock,
					std::chrono::duration<double>(sleep_seconds
						+ std::uniform_real_distribution<double>(0, sleep_seconds_random_part)(rng)));
				continue;
			}

			/// Лучшей задачи не нашлось, а эта задача в прошлый раз ничего не сделала, и поэтому ей назначено некоторое время спать.
			time_t current_time = time(0);
			if (min_time > current_time)
			{
				std::unique_lock<std::mutex> lock(tasks_mutex);
				wake_event.wait_for(lock, std::chrono::duration<double>(
					min_time - current_time + std::uniform_real_distribution<double>(0, sleep_seconds_random_part)(rng)));
			}

			Poco::ScopedReadRWLock rlock(task->rwlock);

			if (task->removed)
				continue;

			{
				CurrentMetrics::Increment metric_increment{CurrentMetrics::BackgroundPoolTask};

				Context context(*this, counters_diff);
				bool done_work = task->function(context);

				/// Если задача сделала полезную работу, то она сможет выполняться в следующий раз хоть сразу.
				/// Если нет - добавляем задержку перед повторным исполнением.
				task->next_time_to_execute = time(0) + (done_work ? 0 : sleep_seconds);
			}
		}
		catch (...)
		{
			has_exception = true;
			tryLogCurrentException(__PRETTY_FUNCTION__);
		}

		/// Вычтем все счётчики обратно.
		if (!counters_diff.empty())
		{
			std::unique_lock<std::mutex> lock(counters_mutex);
			for (const auto & it : counters_diff)
				counters[it.first] -= it.second;
		}

		if (shutdown)
