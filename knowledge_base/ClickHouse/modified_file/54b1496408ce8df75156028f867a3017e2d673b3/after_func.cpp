        return {};
    }

    /// Now task is ready for execution
    return task;
}


static void filterAndSortQueueNodes(Strings & all_nodes)
{
    all_nodes.erase(std::remove_if(all_nodes.begin(), all_nodes.end(), [] (const String & s) { return !startsWith(s, "query-"); }), all_nodes.end());
    std::sort(all_nodes.begin(), all_nodes.end());
}

void DDLWorker::scheduleTasks()
{
    LOG_DEBUG(log, "Scheduling tasks");
    auto zookeeper = tryGetZooKeeper();

    for (auto & task : current_tasks)
    {
        /// Main thread of DDLWorker was restarted, probably due to lost connection with ZooKeeper.
        /// We have some unfinished tasks. To avoid duplication of some queries, try to write execution status.
        if (task->was_executed)
        {
            bool task_still_exists = zookeeper->exists(task->entry_path);
            bool status_written = zookeeper->exists(task->getFinishedNodePath());
            if (!status_written && task_still_exists)
            {
                processTask(*task, zookeeper);
            }
        }
    }

    Strings queue_nodes = zookeeper->getChildren(queue_dir, nullptr, queue_updated_event);
    filterAndSortQueueNodes(queue_nodes);
    if (queue_nodes.empty())
    {
        LOG_TRACE(log, "No tasks to schedule");
        return;
    }
    else if (max_tasks_in_queue < queue_nodes.size())
        cleanup_event->set();

    bool server_startup = current_tasks.empty();
    auto begin_node = queue_nodes.begin();

    if (!server_startup)
    {
        /// We will recheck status of last executed tasks. It's useful if main thread was just restarted.
        auto & min_task = current_tasks.front();
        String min_entry_name = last_skipped_entry_name ? std::min(min_task->entry_name, *last_skipped_entry_name) : min_task->entry_name;
        begin_node = std::upper_bound(queue_nodes.begin(), queue_nodes.end(), min_entry_name);
        current_tasks.remove_if([](const DDLTaskPtr & t) { return t->completely_processed.load(); });
    }

    assert(current_tasks.empty());

    for (auto it = begin_node; it != queue_nodes.end() && !stop_flag; ++it)
    {
        String entry_name = *it;
        LOG_TRACE(log, "Checking task {}", entry_name);

        String reason;
        auto task = initAndCheckTask(entry_name, reason, zookeeper);
        if (!task)
        {
            LOG_DEBUG(log, "Will not execute task {}: {}", entry_name, reason);
            updateMaxDDLEntryID(entry_name);
            last_skipped_entry_name.emplace(entry_name);
            continue;
        }

        auto & saved_task = saveTask(std::move(task));

