
OSLoop *Win32Loop::instance( intf_thread_t *pIntf )
{
    if( pIntf->p_sys->p_osLoop == NULL )
    {
        OSLoop *pOsLoop = new Win32Loop( pIntf );
        pIntf->p_sys->p_osLoop = pOsLoop;
    }
    return pIntf->p_sys->p_osLoop;
}


void Win32Loop::destroy( intf_thread_t *pIntf )
{
    if( pIntf->p_sys->p_osLoop )
    {
        delete pIntf->p_sys->p_osLoop;
        pIntf->p_sys->p_osLoop = NULL;
    }
}


void Win32Loop::run()
{
    MSG msg;

    // Compute windows message list
    while( GetMessage( &msg, NULL, 0, 0 ) )
    {
        Win32Factory *pFactory =
            (Win32Factory*)Win32Factory::instance( getIntf() );
        GenericWindow *pWin = pFactory->m_windowMap[msg.hwnd];
        if( pWin == NULL )
        {
            // We are probably getting a message for a tooltip (which has no
            // associated GenericWindow), for a timer, or for the parent window
            DispatchMessage( &msg );
            continue;
        }

        GenericWindow &win = *pWin;
        switch( msg.message )
        {
            case WM_PAINT:
            {
                PAINTSTRUCT Infos;
                BeginPaint( msg.hwnd, &Infos );
                EvtRefresh evt( getIntf(), 0, 0, Infos.rcPaint.right,
                                Infos.rcPaint.bottom );
                EndPaint( msg.hwnd, &Infos );
                win.processEvent( evt );
                break;
            }
            case WM_MOUSEMOVE:
            {
                // Needed to generate WM_MOUSELEAVE events
                TRACKMOUSEEVENT TrackEvent;
                TrackEvent.cbSize      = sizeof( TRACKMOUSEEVENT );
                TrackEvent.dwFlags     = TME_LEAVE;
                TrackEvent.hwndTrack   = msg.hwnd;
                TrackEvent.dwHoverTime = 1;
                TrackMouseEvent( &TrackEvent );

                // Compute the absolute position of the mouse
                int x = GET_X_LPARAM( msg.lParam ) + win.getLeft();
                int y = GET_Y_LPARAM( msg.lParam ) + win.getTop();
                EvtMotion evt( getIntf(), x, y );
                win.processEvent( evt );
                break;
            }
            case WM_MOUSELEAVE:
            {
                EvtLeave evt( getIntf() );
                win.processEvent( evt );
                break;
            }
            case WM_MOUSEWHEEL:
            {
                int x = GET_X_LPARAM( msg.lParam ) - win.getLeft();
                int y = GET_Y_LPARAM( msg.lParam ) - win.getTop();
                int mod = getMod( msg.wParam );
                if( GET_WHEEL_DELTA_WPARAM( msg.wParam ) > 0 )
                {
                    EvtScroll evt( getIntf(), x, y, EvtScroll::kUp, mod );
                    win.processEvent( evt );
                }
                else
                {
                    EvtScroll evt( getIntf(), x, y, EvtScroll::kDown, mod );
                    win.processEvent( evt );
                }
                break;
            }
            case WM_LBUTTONDOWN:
            {
                SetCapture( msg.hwnd );
                EvtMouse evt( getIntf(), GET_X_LPARAM( msg.lParam ),
                              GET_Y_LPARAM( msg.lParam ), EvtMouse::kLeft,
                              EvtMouse::kDown, getMod( msg.wParam ) );
                win.processEvent( evt );
                break;
            }
            case WM_RBUTTONDOWN:
            {
                SetCapture( msg.hwnd );
                EvtMouse evt( getIntf(), GET_X_LPARAM( msg.lParam ),
                              GET_Y_LPARAM( msg.lParam ), EvtMouse::kRight,
                              EvtMouse::kDown, getMod( msg.wParam ) );
                win.processEvent( evt );
                break;
            }
            case WM_LBUTTONUP:
            {
                ReleaseCapture();
                EvtMouse evt( getIntf(), GET_X_LPARAM( msg.lParam ),
                              GET_Y_LPARAM( msg.lParam ), EvtMouse::kLeft,
                              EvtMouse::kUp, getMod( msg.wParam ) );
                win.processEvent( evt );
                break;
            }
            case WM_RBUTTONUP:
            {
                ReleaseCapture();
                EvtMouse evt( getIntf(), GET_X_LPARAM( msg.lParam ),
                              GET_Y_LPARAM( msg.lParam ), EvtMouse::kRight,
                              EvtMouse::kUp, getMod( msg.wParam ) );
                win.processEvent( evt );
                break;
            }
            case WM_LBUTTONDBLCLK:
            {
                ReleaseCapture();
                EvtMouse evt( getIntf(), GET_X_LPARAM( msg.lParam ),
                              GET_Y_LPARAM( msg.lParam ), EvtMouse::kLeft,
                              EvtMouse::kDblClick, getMod( msg.wParam ) );
                win.processEvent( evt );
                break;
            }
            case WM_RBUTTONDBLCLK:
            {
                ReleaseCapture();
                EvtMouse evt( getIntf(), GET_X_LPARAM( msg.lParam ),
                              GET_Y_LPARAM( msg.lParam ), EvtMouse::kRight,
                              EvtMouse::kDblClick, getMod( msg.wParam ) );
                win.processEvent( evt );
                break;
            }
            case WM_KEYDOWN:
            case WM_SYSKEYDOWN:
            case WM_KEYUP:
            case WM_SYSKEYUP:
            {
                // The key events are first processed here and not translated
                // into WM_CHAR events because we need to know the status of
                // the modifier keys.

                // Get VLC key code from the virtual key code
                int key = virtKeyToVlcKey[msg.wParam];
                if( !key )
                {
                    // This appears to be a "normal" (ascii) key
                    key = tolower( MapVirtualKey( msg.wParam, 2 ) );
                }

                if( key )
                {
                    // Get the modifier
                    int mod = 0;
                    if( GetKeyState( VK_CONTROL ) & 0x8000 )
                    {
                        mod |= EvtInput::kModCtrl;
                    }
                    if( GetKeyState( VK_SHIFT ) & 0x8000 )
                    {
                        mod |= EvtInput::kModShift;
                    }
                    if( GetKeyState( VK_MENU ) & 0x8000 )
                    {
                        mod |= EvtInput::kModAlt;
                    }

                    // Get the state
