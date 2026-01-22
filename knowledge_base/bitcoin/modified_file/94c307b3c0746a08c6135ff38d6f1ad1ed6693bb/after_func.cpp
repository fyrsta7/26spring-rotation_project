    event_base_dispatch(base);
    // Event loop will be interrupted by InterruptHTTPServer()
    LogDebug(BCLog::HTTP, "Exited http event loop\n");
}

/** Bind HTTP server to specified addresses */
static bool HTTPBindAddresses(struct evhttp* http)
{
    uint16_t http_port{static_cast<uint16_t>(gArgs.GetIntArg("-rpcport", BaseParams().RPCPort()))};
    std::vector<std::pair<std::string, uint16_t>> endpoints;

    // Determine what addresses to bind to
    if (!(gArgs.IsArgSet("-rpcallowip") && gArgs.IsArgSet("-rpcbind"))) { // Default to loopback if not allowing external IPs
        endpoints.emplace_back("::1", http_port);
        endpoints.emplace_back("127.0.0.1", http_port);
        if (gArgs.IsArgSet("-rpcallowip")) {
            LogPrintf("WARNING: option -rpcallowip was specified without -rpcbind; this doesn't usually make sense\n");
        }
        if (gArgs.IsArgSet("-rpcbind")) {
            LogPrintf("WARNING: option -rpcbind was ignored because -rpcallowip was not specified, refusing to allow everyone to connect\n");
        }
    } else if (gArgs.IsArgSet("-rpcbind")) { // Specific bind address
        for (const std::string& strRPCBind : gArgs.GetArgs("-rpcbind")) {
            uint16_t port{http_port};
            std::string host;
            SplitHostPort(strRPCBind, port, host);
            endpoints.emplace_back(host, port);
        }
    }

    // Bind addresses
    for (std::vector<std::pair<std::string, uint16_t> >::iterator i = endpoints.begin(); i != endpoints.end(); ++i) {
        LogPrintf("Binding RPC on address %s port %i\n", i->first, i->second);
        evhttp_bound_socket *bind_handle = evhttp_bind_socket_with_handle(http, i->first.empty() ? nullptr : i->first.c_str(), i->second);
        if (bind_handle) {
            const std::optional<CNetAddr> addr{LookupHost(i->first, false)};
            if (i->first.empty() || (addr.has_value() && addr->IsBindAny())) {
                LogPrintf("WARNING: the RPC server is not safe to expose to untrusted networks such as the public internet\n");
            }
            // Set the no-delay option (disable Nagle's algorithm) on the TCP socket.
            evutil_socket_t fd = evhttp_bound_socket_get_fd(bind_handle);
            int one = 1;
            if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (sockopt_arg_type)&one, sizeof(one)) == SOCKET_ERROR) {
                LogInfo("WARNING: Unable to set TCP_NODELAY on RPC server socket, continuing anyway\n");
            }
            boundSockets.push_back(bind_handle);
