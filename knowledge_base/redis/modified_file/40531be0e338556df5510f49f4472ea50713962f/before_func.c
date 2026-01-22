                } else if (t == LUA_TNUMBER) {
                    addReplyLongLong(c,(long long)lua_tonumber(lua,-1));
                    mbulklen++;
                }
                lua_pop(lua,1);
            }
            setDeferredMultiBulkLength(c,replylen,mbulklen);
        }
        break;
    default:
        addReply(c,shared.nullbulk);
    }
    lua_pop(lua,1);
}

/* Set an array of Redis String Objects as a Lua array (table) stored into a
 * global variable. */
void luaSetGlobalArray(lua_State *lua, char *var, robj **elev, int elec) {
    int j;

    lua_newtable(lua);
    for (j = 0; j < elec; j++) {
        lua_pushlstring(lua,(char*)elev[j]->ptr,sdslen(elev[j]->ptr));
        lua_rawseti(lua,-2,j+1);
    }
    lua_setglobal(lua,var);
}

void evalCommand(redisClient *c) {
    lua_State *lua = server.lua;
    char funcname[43];
    long long numkeys;

    /* Get the number of arguments that are keys */
    if (getLongLongFromObjectOrReply(c,c->argv[2],&numkeys,NULL) != REDIS_OK)
        return;
    if (numkeys > (c->argc - 3)) {
        addReplyError(c,"Number of keys can't be greater than number of args");
        return;
    }

    /* We obtain the script SHA1, then check if this function is already
     * defined into the Lua state */
    funcname[0] = 'f';
    funcname[1] = '_';
    hashScript(funcname+2,c->argv[1]->ptr,sdslen(c->argv[1]->ptr));
    lua_getglobal(lua, funcname);
    if (lua_isnil(lua,1)) {
        /* Function not defined... let's define it. */
        sds funcdef = sdsempty();

        lua_pop(lua,1); /* remove the nil from the stack */
        funcdef = sdscat(funcdef,"function ");
        funcdef = sdscatlen(funcdef,funcname,42);
        funcdef = sdscatlen(funcdef," ()\n",4);
        funcdef = sdscatlen(funcdef,c->argv[1]->ptr,sdslen(c->argv[1]->ptr));
        funcdef = sdscatlen(funcdef,"\nend\n",5);
        /* printf("Defining:\n%s\n",funcdef); */

        if (luaL_loadbuffer(lua,funcdef,sdslen(funcdef),"func definition")) {
            addReplyErrorFormat(c,"Error compiling script (new function): %s\n",
                lua_tostring(lua,-1));
            lua_pop(lua,1);
            sdsfree(funcdef);
            return;
