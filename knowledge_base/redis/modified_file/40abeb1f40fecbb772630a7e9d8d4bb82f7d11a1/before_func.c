    return p;
}

void luaPushError(lua_State *lua, char *error) {
    lua_Debug dbg;

    lua_newtable(lua);
    lua_pushstring(lua,"err");

    /* Attempt to figure out where this function was called, if possible */
    if(lua_getstack(lua, 1, &dbg) && lua_getinfo(lua, "nSl", &dbg)) {
        sds msg = sdscatprintf(sdsempty(), "%s: %d: %s",
            dbg.source, dbg.currentline, error);
        lua_pushstring(lua, msg);
        sdsfree(msg);
    } else {
        lua_pushstring(lua, error);
    }
    lua_settable(lua,-3);
}

/* Sort the array currently in the stack. We do this to make the output
 * of commands like KEYS or SMEMBERS something deterministic when called
 * from Lua (to play well with AOf/replication).
 *
 * The array is sorted using table.sort itself, and assuming all the
 * list elements are strings. */
void luaSortArray(lua_State *lua) {
    /* Initial Stack: array */
    lua_getglobal(lua,"table");
    lua_pushstring(lua,"sort");
    lua_gettable(lua,-2);       /* Stack: array, table, table.sort */
    lua_pushvalue(lua,-3);      /* Stack: array, table, table.sort, array */
    if (lua_pcall(lua,1,0,0)) {
        /* Stack: array, table, error */

        /* We are not interested in the error, we assume that the problem is
         * that there are 'false' elements inside the array, so we try
         * again with a slower function but able to handle this case, that
         * is: table.sort(table, __redis__compare_helper) */
        lua_pop(lua,1);             /* Stack: array, table */
        lua_pushstring(lua,"sort"); /* Stack: array, table, sort */
        lua_gettable(lua,-2);       /* Stack: array, table, table.sort */
        lua_pushvalue(lua,-3);      /* Stack: array, table, table.sort, array */
        lua_getglobal(lua,"__redis__compare_helper");
        /* Stack: array, table, table.sort, array, __redis__compare_helper */
        lua_call(lua,2,0);
    }
    /* Stack: array (sorted), table */
    lua_pop(lua,1);             /* Stack: array (sorted) */
}

int luaRedisGenericCommand(lua_State *lua, int raise_error) {
    int j, argc = lua_gettop(lua);
    struct redisCommand *cmd;
    robj **argv;
    redisClient *c = server.lua_client;
    sds reply;

    /* Require at least one argument */
    if (argc == 0) {
        luaPushError(lua,
            "Please specify at least one argument for redis.call()");
        return 1;
    }

    /* Build the arguments vector */
    argv = zmalloc(sizeof(robj*)*argc);
    for (j = 0; j < argc; j++) {
        if (!lua_isstring(lua,j+1)) break;
        argv[j] = createStringObject((char*)lua_tostring(lua,j+1),
                                     lua_strlen(lua,j+1));
    }
    
    /* Check if one of the arguments passed by the Lua script
     * is not a string or an integer (lua_isstring() return true for
     * integers as well). */
    if (j != argc) {
        j--;
        while (j >= 0) {
            decrRefCount(argv[j]);
            j--;
        }
        zfree(argv);
        luaPushError(lua,
            "Lua redis() command arguments must be strings or integers");
        return 1;
    }

    /* Setup our fake client for command execution */
    c->argv = argv;
    c->argc = argc;

    /* Command lookup */
    cmd = lookupCommand(argv[0]->ptr);
    if (!cmd || ((cmd->arity > 0 && cmd->arity != argc) ||
                   (argc < -cmd->arity)))
    {
        if (cmd)
            luaPushError(lua,
                "Wrong number of args calling Redis command From Lua script");
        else
            luaPushError(lua,"Unknown Redis command called from Lua script");
        goto cleanup;
    }

    /* There are commands that are not allowed inside scripts. */
    if (cmd->flags & REDIS_CMD_NOSCRIPT) {
        luaPushError(lua, "This Redis command is not allowed from scripts");
        goto cleanup;
    }

    /* Write commands are forbidden against read-only slaves, or if a
     * command marked as non-deterministic was already called in the context
     * of this script. */
    if (cmd->flags & REDIS_CMD_WRITE) {
        if (server.lua_random_dirty) {
            luaPushError(lua,
                "Write commands not allowed after non deterministic commands");
            goto cleanup;
        } else if (server.masterhost && server.repl_slave_ro &&
                   !server.loading &&
                   !(server.lua_caller->flags & REDIS_MASTER))
        {
            luaPushError(lua, shared.roslaveerr->ptr);
            goto cleanup;
        } else if (server.stop_writes_on_bgsave_err &&
                   server.saveparamslen > 0 &&
                   server.lastbgsave_status == REDIS_ERR)
        {
            luaPushError(lua, shared.bgsaveerr->ptr);
            goto cleanup;
