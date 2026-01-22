  PatBlt( hdc, rect->left, rect->top + height, width,
	  rect->bottom - rect->top - height, PATINVERT );
  PatBlt( hdc, rect->left + width, rect->bottom - 1,
	  rect->right - rect->left - width, -height, PATINVERT );
  PatBlt( hdc, rect->right - 1, rect->top, -width,
	  rect->bottom - rect->top - height, PATINVERT );
  SelectObject( hdc, hbrush );
}

VOID STATIC
UserDrawMovingFrame(HDC hdc, RECT *rect, BOOL thickframe)
{
  if(thickframe)
  {
    UserDrawWindowFrame(hdc, rect, GetSystemMetrics(SM_CXFRAME), GetSystemMetrics(SM_CYFRAME));
  }
  else
  {
    UserDrawWindowFrame(hdc, rect, 1, 1);
  }
}

VOID STATIC
DefWndDoSizeMove(HWND hwnd, WORD wParam)
{
  HRGN DesktopRgn;
  MSG msg;
  RECT sizingRect, mouseRect, origRect, clipRect, unmodRect;
  HDC hdc;
  LONG hittest = (LONG)(wParam & 0x0f);
  HCURSOR hDragCursor = 0, hOldCursor = 0;
  POINT minTrack, maxTrack;
  POINT capturePoint, pt;
  ULONG Style = GetWindowLongW(hwnd, GWL_STYLE);
  ULONG ExStyle = GetWindowLongW(hwnd, GWL_EXSTYLE); 
  BOOL thickframe;
  BOOL iconic = Style & WS_MINIMIZE;
  BOOL moved = FALSE;
  DWORD dwPoint = GetMessagePos();
  BOOL DragFullWindows = FALSE;
  HWND hWndParent = NULL;

  SystemParametersInfoA(SPI_GETDRAGFULLWINDOWS, 0, &DragFullWindows, 0);
  
  pt.x = GET_X_LPARAM(dwPoint);
  pt.y = GET_Y_LPARAM(dwPoint);
  capturePoint = pt;
  
  if (IsZoomed(hwnd) || !IsWindowVisible(hwnd))
    {
      return;
    }
  
  thickframe = UserHasThickFrameStyle(Style, ExStyle) && !(Style & WS_MINIMIZE);
  if ((wParam & 0xfff0) == SC_MOVE)
    {
      if (!hittest) 
	{
	  hittest = DefWndStartSizeMove(hwnd, wParam, &capturePoint);
	}
      if (!hittest)
	{
	  return;
	}
    }
  else  /* SC_SIZE */
    {
      if (!thickframe)
	{
	  return;
	}
      if (hittest && ((wParam & 0xfff0) != SC_MOUSEMENU))
	{
          hittest += (HTLEFT - WMSZ_LEFT);
	}
      else
	{
	  SetCapture(hwnd);
	  hittest = DefWndStartSizeMove(hwnd, wParam, &capturePoint);
	  if (!hittest)
	    {
	      ReleaseCapture();
	      return;
	    }
	}
    }

  if (Style & WS_CHILD)
    {
      hWndParent = GetParent(hwnd);
    }
  
  /* Get min/max info */
  
  WinPosGetMinMaxInfo(hwnd, NULL, NULL, &minTrack, &maxTrack);
  GetWindowRect(hwnd, &sizingRect);
  unmodRect = sizingRect;
  if (Style & WS_CHILD)
    {
      MapWindowPoints( 0, hWndParent, (LPPOINT)&sizingRect, 2 );
      GetClientRect(hWndParent, &mouseRect );
      clipRect = mouseRect;
      MapWindowPoints(hWndParent, HWND_DESKTOP, (LPPOINT)&clipRect, 2);
    }
  else 
    {
      if(!(ExStyle & WS_EX_TOPMOST))
      {
        SystemParametersInfoW(SPI_GETWORKAREA, 0, &clipRect, 0);
        mouseRect = clipRect;
      }
      else
      {
        SetRect(&mouseRect, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
        clipRect = mouseRect;
      }
    }
  ClipCursor(&clipRect);
  
  origRect = sizingRect;
  if (ON_LEFT_BORDER(hittest))
    {
      mouseRect.left  = max( mouseRect.left, sizingRect.right-maxTrack.x );
      mouseRect.right = min( mouseRect.right, sizingRect.right-minTrack.x );
    }
  else if (ON_RIGHT_BORDER(hittest))
    {
      mouseRect.left  = max( mouseRect.left, sizingRect.left+minTrack.x );
      mouseRect.right = min( mouseRect.right, sizingRect.left+maxTrack.x );
    }
  if (ON_TOP_BORDER(hittest))
    {
      mouseRect.top    = max( mouseRect.top, sizingRect.bottom-maxTrack.y );
      mouseRect.bottom = min( mouseRect.bottom,sizingRect.bottom-minTrack.y);
    }
  else if (ON_BOTTOM_BORDER(hittest))
    {
      mouseRect.top    = max( mouseRect.top, sizingRect.top+minTrack.y );
      mouseRect.bottom = min( mouseRect.bottom, sizingRect.top+maxTrack.y );
    }
  if (Style & WS_CHILD)
    {      
      MapWindowPoints( hWndParent, 0, (LPPOINT)&mouseRect, 2 );
    }
  
  SendMessageA( hwnd, WM_ENTERSIZEMOVE, 0, 0 );
  NtUserSetGUIThreadHandle(MSQ_STATE_MOVESIZE, hwnd);
  if (GetCapture() != hwnd) SetCapture( hwnd );    
  
  if (Style & WS_CHILD)
    {
      /* Retrieve a default cache DC (without using the window style) */
      hdc = GetDCEx(hWndParent, 0, DCX_CACHE);
      DesktopRgn = NULL;
    }
  else
    {
      hdc = GetDC( 0 );
      DesktopRgn = CreateRectRgnIndirect(&clipRect);
    }
  
  SelectObject(hdc, DesktopRgn);
  
  if( iconic ) /* create a cursor for dragging */
    {
      HICON hIcon = (HICON)GetClassLongW(hwnd, GCL_HICON);
      if(!hIcon) hIcon = (HICON)SendMessageW( hwnd, WM_QUERYDRAGICON, 0, 0L);
      if( hIcon ) hDragCursor = CursorIconToCursor( hIcon, TRUE );
      if( !hDragCursor ) iconic = FALSE;
    }
  
  /* invert frame if WIN31_LOOK to indicate mouse click on caption */
  if( !iconic && !DragFullWindows)
    {
      UserDrawMovingFrame( hdc, &sizingRect, thickframe);
    }
  
  for(;;)
    {
      int dx = 0, dy = 0;

      GetMessageW(&msg, 0, 0, 0);
      
      /* Exit on button-up, Return, or Esc */
      if ((msg.message == WM_LBUTTONUP) ||
	  ((msg.message == WM_KEYDOWN) && 
	   ((msg.wParam == VK_RETURN) || (msg.wParam == VK_ESCAPE)))) break;
      
      if (msg.message == WM_PAINT)
        {
	  if(!iconic && !DragFullWindows) UserDrawMovingFrame( hdc, &sizingRect, thickframe );
	  UpdateWindow( msg.hwnd );
	  if(!iconic && !DragFullWindows) UserDrawMovingFrame( hdc, &sizingRect, thickframe );
	  continue;
        }
      
      if ((msg.message != WM_KEYDOWN) && (msg.message != WM_MOUSEMOVE))
	continue;  /* We are not interested in other messages */
      
      pt = msg.pt;
      
      if (msg.message == WM_KEYDOWN) switch(msg.wParam)
	{
	case VK_UP:    pt.y -= 8; break;
	case VK_DOWN:  pt.y += 8; break;
	case VK_LEFT:  pt.x -= 8; break;
	case VK_RIGHT: pt.x += 8; break;		
	}
      
      pt.x = max( pt.x, mouseRect.left );
      pt.x = min( pt.x, mouseRect.right );
      pt.y = max( pt.y, mouseRect.top );
      pt.y = min( pt.y, mouseRect.bottom );
      
      dx = pt.x - capturePoint.x;
      dy = pt.y - capturePoint.y;
      
      if (dx || dy)
	{
	  if( !moved )
	    {
	      moved = TRUE;
	      
		if( iconic ) /* ok, no system popup tracking */
		  {
		    hOldCursor = SetCursor(hDragCursor);
		    ShowCursor( TRUE );
		  } 
	    }
	  
	  if (msg.message == WM_KEYDOWN) SetCursorPos( pt.x, pt.y );
	  else
	    {
	      RECT newRect = unmodRect;
	      WPARAM wpSizingHit = 0;
	      
	      if (hittest == HTCAPTION) OffsetRect( &newRect, dx, dy );
	      if (ON_LEFT_BORDER(hittest)) newRect.left += dx;
	      else if (ON_RIGHT_BORDER(hittest)) newRect.right += dx;
	      if (ON_TOP_BORDER(hittest)) newRect.top += dy;
	      else if (ON_BOTTOM_BORDER(hittest)) newRect.bottom += dy;
	      if(!iconic && !DragFullWindows) UserDrawMovingFrame( hdc, &sizingRect, thickframe );
	      capturePoint = pt;
	      
	      /* determine the hit location */
	      if (hittest >= HTLEFT && hittest <= HTBOTTOMRIGHT)
		wpSizingHit = WMSZ_LEFT + (hittest - HTLEFT);
	      unmodRect	= newRect;
	      SendMessageA( hwnd, WM_SIZING, wpSizingHit, (LPARAM)&newRect );
	      
	      if (!iconic)
		{
		  if(!DragFullWindows)
		    UserDrawMovingFrame( hdc, &newRect, thickframe );
		  else {
		    /* To avoid any deadlocks, all the locks on the windows
		       structures must be suspended before the SetWindowPos */
		    SetWindowPos( hwnd, 0, newRect.left, newRect.top,
				  newRect.right - newRect.left,
				  newRect.bottom - newRect.top,
				  ( hittest == HTCAPTION ) ? SWP_NOSIZE : 0 );
		  }
		}
	      sizingRect = newRect;
	    }
	}
    }
  
  ReleaseCapture();
  ClipCursor(NULL);
  if( iconic )
    {
      if( moved ) /* restore cursors, show icon title later on */
	{
	  ShowCursor( FALSE );
	  SetCursor( hOldCursor );
	}
      DestroyCursor( hDragCursor );
    }
  else if(!DragFullWindows)
      UserDrawMovingFrame( hdc, &sizingRect, thickframe );
  
  if (Style & WS_CHILD)
    ReleaseDC( hWndParent, hdc );
  else
  {
    ReleaseDC( 0, hdc );
    if(DesktopRgn)
    {
      DeleteObject(DesktopRgn);
    }
  }
  NtUserSetGUIThreadHandle(MSQ_STATE_MOVESIZE, NULL);
  SendMessageA( hwnd, WM_EXITSIZEMOVE, 0, 0 );
  SendMessageA( hwnd, WM_SETVISIBLE, !IsIconic(hwnd), 0L);
  
  /* window moved or resized */
  if (moved)
    {
      /* if the moving/resizing isn't canceled call SetWindowPos
       * with the new position or the new size of the window
       */
      if (!((msg.message == WM_KEYDOWN) && (msg.wParam == VK_ESCAPE)) )
        {
	  /* NOTE: SWP_NOACTIVATE prevents document window activation in Word 6 */
	  if(!DragFullWindows)
	    SetWindowPos( hwnd, 0, sizingRect.left, sizingRect.top,
			  sizingRect.right - sizingRect.left,
