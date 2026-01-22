bool MozJSImplScope::_interruptCallback(JSContext* cx) {
    auto scope = getScope(cx);

    JS_SetInterruptCallback(scope->_runtime, nullptr);
    auto guard = MakeGuard([&]() { JS_SetInterruptCallback(scope->_runtime, _interruptCallback); });

    if (scope->_pendingGC.load()) {
        scope->_pendingGC.store(false);
        JS_GC(scope->_runtime);
    } else {
        JS_MaybeGC(cx);
    }

    if (scope->_hasOutOfMemoryException) {
        scope->_status = Status(ErrorCodes::JSInterpreterFailure, "Out of memory");
    } else if (scope->isKillPending()) {
        scope->_status = Status(ErrorCodes::JSInterpreterFailure, "Interrupted by the host");
    }

    if (!scope->_status.isOK()) {
        scope->_engine->getDeadlineMonitor().stopDeadline(scope);
        scope->unregisterOperation();
    }

    return scope->_status.isOK();
}

void MozJSImplScope::_gcCallback(JSRuntime* rt, JSGCStatus status, void* data) {
    if (!shouldLog(logger::LogSeverity::Debug(1))) {
        // don't collect stats unless verbose
        return;
    }

    log() << "MozJS GC " << (status == JSGC_BEGIN ? "prologue" : "epilogue") << " heap stats - "
          << " total: " << mongo::sm::get_total_bytes() << " limit: " << mongo::sm::get_max_bytes()
          << std::endl;
}

MozJSImplScope::MozRuntime::MozRuntime(const MozJSScriptEngine* engine) {
    mongo::sm::reset(kMallocMemoryLimit);

    // If this runtime isn't running on an NSPR thread, then it is
    // running on a mongo thread. In that case, we need to insert a
    // fake NSPR thread so that the SM runtime can call PR functions
    // without falling over.
    auto thread = PR_GetCurrentThread();
    if (!thread) {
        PR_BindThread(_thread = PR_CreateFakeThread());
    }

    {
        stdx::unique_lock<stdx::mutex> lk(gRuntimeCreationMutex);

        if (gFirstRuntimeCreated) {
            // If we've already made a runtime, just proceed
            lk.unlock();
        } else {
            // If this is the first one, hold the lock until after the first
            // one's done
            gFirstRuntimeCreated = true;
        }
