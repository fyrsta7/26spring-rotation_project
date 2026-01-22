DWORD WINAPI RenderThread::_ThreadProc()
{
    while (_fKeepRunning)
    {
        WaitForSingleObject(_hPaintEnabledEvent, INFINITE);

        if (!_fNextFrameRequested.exchange(false, std::memory_order_acq_rel))
        {
            // <--
            // If `NotifyPaint` is called at this point, then it will not
            // set the event because `_fWaiting` is not `true` yet so we have
            // to check again below.

            _fWaiting.store(true, std::memory_order_release);

            // check again now (see comment above)
            if (!_fNextFrameRequested.exchange(false, std::memory_order_acq_rel))
            {
                // Wait until a next frame is requested.
                WaitForSingleObject(_hEvent, INFINITE);
            }

            // <--
            // If `NotifyPaint` is called at this point, then it _will_ set
            // the event because `_fWaiting` is `true`, but we're not waiting
            // anymore!
            // This can probably happen quite often: imagine a scenario where
            // we are waiting, and the terminal calls `NotifyPaint` twice
            // very quickly.
            // In that case, both calls might end up calling `SetEvent`. The
            // first one will resume this thread and the second one will
            // `SetEvent` the event. So the next time we wait, the event will
            // already be set and we won't actually wait.
            // Because it can happen often, and because rendering is an
            // expensive operation, we should reset the event to not render
            // again if nothing changed.

            _fWaiting.store(false, std::memory_order_release);

            // see comment above
            ResetEvent(_hEvent);
        }

        ResetEvent(_hPaintCompletedEvent);

        _pRenderer->WaitUntilCanRender();
        LOG_IF_FAILED(_pRenderer->PaintFrame());

        SetEvent(_hPaintCompletedEvent);
    }

    return S_OK;
}
