
/******************************************************************************
  *     SetDataFormat : the application can choose the format of the data
  *   the device driver sends back with GetDeviceState.
  *
  *   For the moment, only the "standard" configuration (c_dfDIMouse) is supported
  *   in absolute and relative mode.
  */
static HRESULT WINAPI SysMouseAImpl_SetDataFormat(
	LPDIRECTINPUTDEVICE8A iface,LPCDIDATAFORMAT df
)
{
    SysMouseImpl *This = (SysMouseImpl *)iface;
    
    TRACE("(this=%p,%p)\n",This,df);
    
    _dump_DIDATAFORMAT(df);
    
    /* Tests under windows show that a call to SetDataFormat always sets the mouse
       in relative mode whatever the dwFlags value (DIDF_ABSAXIS/DIDF_RELAXIS).
       To switch in absolute mode, SetProperty must be used. */
    This->absolute = 0;
    
    /* Store the new data format */
    This->df = HeapAlloc(GetProcessHeap(),0,df->dwSize);
    memcpy(This->df, df, df->dwSize);
    This->df->rgodf = HeapAlloc(GetProcessHeap(),0,df->dwNumObjs*df->dwObjSize);
    memcpy(This->df->rgodf,df->rgodf,df->dwNumObjs*df->dwObjSize);
    
    /* Prepare all the data-conversion filters */
    This->wine_df = create_DataFormat(&(Wine_InternalMouseFormat), df, This->offset_array);
    
    return DI_OK;
}

/* low-level mouse hook */
static LRESULT CALLBACK dinput_mouse_hook( int code, WPARAM wparam, LPARAM lparam )
{
    LRESULT ret;
    MSLLHOOKSTRUCT *hook = (MSLLHOOKSTRUCT *)lparam;
    SysMouseImpl* This = (SysMouseImpl*) current_lock;
    DWORD dwCoop;
    static long last_event = 0;
    int wdata;

    if (code != HC_ACTION) return CallNextHookEx( This->hook, code, wparam, lparam );

    EnterCriticalSection(&(This->crit));
    dwCoop = This->dwCoopLevel;

    /* Only allow mouse events every 10 ms.
     * This is to allow the cursor to start acceleration before
     * the warps happen. But if it involves a mouse button event we
     * allow it since we don't want to lose the clicks.
     */
    if (((GetCurrentTime() - last_event) < 10)
        && wparam == WM_MOUSEMOVE)
	goto end;
    else last_event = GetCurrentTime();
    
    /* Mouse moved -> send event if asked */
    if (This->hEvent)
        SetEvent(This->hEvent);
    
    if (wparam == WM_MOUSEMOVE) {
	if (This->absolute) {
	    if (hook->pt.x != This->prevX)
		GEN_EVENT(This->offset_array[WINE_MOUSE_X_POSITION], hook->pt.x, hook->time, 0);
	    if (hook->pt.y != This->prevY)
		GEN_EVENT(This->offset_array[WINE_MOUSE_Y_POSITION], hook->pt.y, hook->time, 0);
	} else {
	    /* Now, warp handling */
	    if ((This->need_warp == WARP_STARTED) &&
		(hook->pt.x == This->mapped_center.x) && (hook->pt.y == This->mapped_center.y)) {
		/* Warp has been done... */
		This->need_warp = WARP_DONE;
		goto end;
	    }
	    
	    /* Relative mouse input with absolute mouse event : the real fun starts here... */
	    if ((This->need_warp == WARP_NEEDED) ||
		(This->need_warp == WARP_STARTED)) {
		if (hook->pt.x != This->prevX)
		    GEN_EVENT(This->offset_array[WINE_MOUSE_X_POSITION], hook->pt.x - This->prevX,
			      hook->time, (This->dinput->evsequence)++);
		if (hook->pt.y != This->prevY)
		    GEN_EVENT(This->offset_array[WINE_MOUSE_Y_POSITION], hook->pt.y - This->prevY,
			      hook->time, (This->dinput->evsequence)++);
	    } else {
		/* This is the first time the event handler has been called after a
		   GetDeviceData or GetDeviceState. */
		if (hook->pt.x != This->mapped_center.x) {
		    GEN_EVENT(This->offset_array[WINE_MOUSE_X_POSITION], hook->pt.x - This->mapped_center.x,
			      hook->time, (This->dinput->evsequence)++);
		    This->need_warp = WARP_NEEDED;
		}
		
		if (hook->pt.y != This->mapped_center.y) {
		    GEN_EVENT(This->offset_array[WINE_MOUSE_Y_POSITION], hook->pt.y - This->mapped_center.y,
			      hook->time, (This->dinput->evsequence)++);
		    This->need_warp = WARP_NEEDED;
		}
	    }
	}
	
	This->prevX = hook->pt.x;
	This->prevY = hook->pt.y;
	
	if (This->absolute) {
	    This->m_state.lX = hook->pt.x;
	    This->m_state.lY = hook->pt.y;
	} else {
	    This->m_state.lX = hook->pt.x - This->mapped_center.x;
	    This->m_state.lY = hook->pt.y - This->mapped_center.y;
	}
    }
    
    TRACE(" msg %x pt %ld %ld (W=%d)\n",
          wparam, hook->pt.x, hook->pt.y, (!This->absolute) && This->need_warp );
    
    switch(wparam) {
        case WM_LBUTTONDOWN:
	    GEN_EVENT(This->offset_array[WINE_MOUSE_L_POSITION], 0x80,
		      hook->time, This->dinput->evsequence++);
	    This->m_state.rgbButtons[0] = 0x80;
	    break;
	case WM_LBUTTONUP:
	    GEN_EVENT(This->offset_array[WINE_MOUSE_L_POSITION], 0x00,
		      hook->time, This->dinput->evsequence++);
	    This->m_state.rgbButtons[0] = 0x00;
	    break;
	case WM_RBUTTONDOWN:
	    GEN_EVENT(This->offset_array[WINE_MOUSE_R_POSITION], 0x80,
		      hook->time, This->dinput->evsequence++);
