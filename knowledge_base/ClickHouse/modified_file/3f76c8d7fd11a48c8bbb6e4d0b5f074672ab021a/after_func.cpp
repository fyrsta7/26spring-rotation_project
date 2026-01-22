ColumnsDescription getStructureOfRemoteTable(
    const Cluster & cluster,
    const StorageID & table_id,
    ContextPtr context,
    const ASTPtr & table_func_ptr)
{
    const auto & shards_info = cluster.getShardsInfo();

    std::string fail_messages;
    
    // use local shard as first priority, as it needs no network communication
    for (const auto & shard_info : shards_info)
    {
        if(shard_info.isLocal()){
            const auto & res = getStructureOfRemoteTableInShard(cluster, shard_info, table_id, context, table_func_ptr);
            if (res.empty())
                continue;

            return res;
        }
    }

    for (const auto & shard_info : shards_info)
    {
        try
        {
            const auto & res = getStructureOfRemoteTableInShard(cluster, shard_info, table_id, context, table_func_ptr);

            /// Expect at least some columns.
            /// This is a hack to handle the empty block case returned by Connection when skip_unavailable_shards is set.
            if (res.empty())
                continue;

            return res;
        }
        catch (const NetException &)
        {
            std::string fail_message = getCurrentExceptionMessage(false);
            fail_messages += fail_message + '\n';
            continue;
        }
    }

    throw NetException(
        "All attempts to get table structure failed. Log: \n\n" + fail_messages + "\n",
        ErrorCodes::NO_REMOTE_SHARD_AVAILABLE);
}
