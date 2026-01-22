        info->rgstate[0] |= STATE_SYSTEM_UNAVAILABLE;
    
    pressed = ((nBar == SB_VERT) == SCROLL_trackVertical && GetCapture() == hwnd);
    
    /* Top/left arrow button state. MSDN says top/right, but I don't believe it */
    info->rgstate[1] = 0;
    if (pressed && SCROLL_trackHitTest == SCROLL_TOP_ARROW)
        info->rgstate[1] |= STATE_SYSTEM_PRESSED;
    if (infoPtr->flags & ESB_DISABLE_LTUP)
        info->rgstate[1] |= STATE_SYSTEM_UNAVAILABLE;

    /* Page up/left region state. MSDN says up/right, but I don't believe it */
    info->rgstate[2] = 0;
    if (infoPtr->curVal == infoPtr->minVal)
        info->rgstate[2] |= STATE_SYSTEM_INVISIBLE;
    if (pressed && SCROLL_trackHitTest == SCROLL_TOP_RECT)
        info->rgstate[2] |= STATE_SYSTEM_PRESSED;

    /* Thumb state */
    info->rgstate[3] = 0;
    if (pressed && SCROLL_trackHitTest == SCROLL_THUMB)
        info->rgstate[3] |= STATE_SYSTEM_PRESSED;

    /* Page down/right region state. MSDN says down/left, but I don't believe it */
    info->rgstate[4] = 0;
    if (infoPtr->curVal >= infoPtr->maxVal - 1)
        info->rgstate[4] |= STATE_SYSTEM_INVISIBLE;
    if (pressed && SCROLL_trackHitTest == SCROLL_BOTTOM_RECT)
        info->rgstate[4] |= STATE_SYSTEM_PRESSED;
    
    /* Bottom/right arrow button state. MSDN says bottom/left, but I don't believe it */
    info->rgstate[5] = 0;
    if (pressed && SCROLL_trackHitTest == SCROLL_BOTTOM_ARROW)
        info->rgstate[5] |= STATE_SYSTEM_PRESSED;
    if (infoPtr->flags & ESB_DISABLE_RTDN)
        info->rgstate[5] |= STATE_SYSTEM_UNAVAILABLE;
        
    return TRUE;
}
#endif
static DWORD FASTCALL
co_IntSetScrollInfo(PWND Window, INT nBar, LPCSCROLLINFO lpsi, BOOL bRedraw)
{
   /*
    * Update the scrollbar state and set action flags according to
    * what has to be done graphics wise.
    */

   LPSCROLLINFO Info;
   PSCROLLBARINFO psbi;
   UINT new_flags;
   INT action = 0;
   PSBDATA pSBData;
   DWORD OldPos = 0;
   BOOL bChangeParams = FALSE; /* Don't show/hide scrollbar if params don't change */
   UINT MaxPage;
   int MaxPos;

   ASSERT_REFS_CO(Window);

   if(!SBID_IS_VALID(nBar))
   {
      EngSetLastError(ERROR_INVALID_PARAMETER);
      ERR("Trying to set scrollinfo for unknown scrollbar type %d", nBar);
      return FALSE;
   }

   if(!co_IntCreateScrollBars(Window))
   {
      return FALSE;
   }

   if (lpsi->cbSize != sizeof(SCROLLINFO) &&
         lpsi->cbSize != (sizeof(SCROLLINFO) - sizeof(lpsi->nTrackPos)))
   {
      EngSetLastError(ERROR_INVALID_PARAMETER);
      return 0;
   }
   if ((lpsi->fMask & ~SIF_THEMED) & ~(SIF_ALL | SIF_DISABLENOSCROLL | SIF_PREVIOUSPOS))
   {
      EngSetLastError(ERROR_INVALID_PARAMETER);
      return 0;
   }

   psbi = IntGetScrollbarInfoFromWindow(Window, nBar);
   Info = IntGetScrollInfoFromWindow(Window, nBar);
   pSBData = IntGetSBData(Window, nBar);

   /* Set the page size */
   if (lpsi->fMask & SIF_PAGE)
   {
      if (Info->nPage != lpsi->nPage)
      {
         Info->nPage = lpsi->nPage;
         pSBData->page = lpsi->nPage;
         bChangeParams = TRUE;
      }
   }

   /* Set the scroll pos */
   if (lpsi->fMask & SIF_POS)
   {
      if (Info->nPos != lpsi->nPos)
      {
         OldPos = Info->nPos;
         Info->nPos = lpsi->nPos;
         pSBData->pos = lpsi->nPos;
      }
   }

   /* Set the scroll range */
   if (lpsi->fMask & SIF_RANGE)
   {
      if (lpsi->nMin > lpsi->nMax)
      {
         Info->nMin = lpsi->nMin;
         Info->nMax = lpsi->nMin;
         pSBData->posMin = lpsi->nMin;
         pSBData->posMax = lpsi->nMin;
         bChangeParams = TRUE;
      }
      else if (Info->nMin != lpsi->nMin || Info->nMax != lpsi->nMax)
      {
         Info->nMin = lpsi->nMin;
         Info->nMax = lpsi->nMax;
         pSBData->posMin = lpsi->nMin;
         pSBData->posMax = lpsi->nMax;
         bChangeParams = TRUE;
      }
   }

   /* Make sure the page size is valid */
   MaxPage = abs(Info->nMax - Info->nMin) + 1;
   if (Info->nPage > MaxPage)
   {
      pSBData->page = Info->nPage = MaxPage;
   }

   /* Make sure the pos is inside the range */
   MaxPos = Info->nMax + 1 - (int)max(Info->nPage, 1);
   ASSERT(MaxPos >= Info->nMin);
   if (Info->nPos < Info->nMin)
   {
      pSBData->pos = Info->nPos = Info->nMin;
   }
   else if (Info->nPos > MaxPos)
   {
      pSBData->pos = Info->nPos = MaxPos;
   }

   /*
    * Don't change the scrollbar state if SetScrollInfo is just called
    * with SIF_DISABLENOSCROLL
    */
   if (!(lpsi->fMask & SIF_ALL))
   {
      //goto done;
      return lpsi->fMask & SIF_PREVIOUSPOS ? OldPos : pSBData->pos;
   }

   /* Check if the scrollbar should be hidden or disabled */
   if (lpsi->fMask & (SIF_RANGE | SIF_PAGE | SIF_DISABLENOSCROLL))
   {
      new_flags = Window->pSBInfo->WSBflags;
      if (Info->nMin + (int)max(Info->nPage, 1) > Info->nMax)
      {
         /* Hide or disable scroll-bar */
         if (lpsi->fMask & SIF_DISABLENOSCROLL)
         {
            new_flags = ESB_DISABLE_BOTH;
            bChangeParams = TRUE;
         }
         else if ((nBar != SB_CTL) && bChangeParams)
         {
            action = SA_SSI_HIDE;
         }
      }
      else /* Show and enable scroll-bar only if no page only changed. */
      if ((lpsi->fMask & ~SIF_THEMED) != SIF_PAGE)
      {
         if ((nBar != SB_CTL) && bChangeParams)
         {
            new_flags = ESB_ENABLE_BOTH;
            action |= SA_SSI_SHOW;
         }
         else if (nBar == SB_CTL)
         {
            new_flags = ESB_ENABLE_BOTH;
         }
      }

      if (Window->pSBInfo->WSBflags != new_flags) /* Check arrow flags */
      {
         Window->pSBInfo->WSBflags = new_flags;
         action |= SA_SSI_REPAINT_ARROWS;
      }
   }

//done:
   if ( action & SA_SSI_HIDE )
   {
      co_UserShowScrollBar(Window, nBar, FALSE, FALSE);
   }
   else
   {
      if ( action & SA_SSI_SHOW )
         if ( co_UserShowScrollBar(Window, nBar, TRUE, TRUE) )
            return lpsi->fMask & SIF_PREVIOUSPOS ? OldPos : pSBData->pos; /* SetWindowPos() already did the painting */
      if (bRedraw)
      {
         if (!(lpsi->fMask & SIF_THEMED)) /* Not Using Themes */
         {
            TRACE("Not using themes.\n");
            if (action & SA_SSI_REPAINT_ARROWS)
            {
               // Redraw the entire bar.
               RECTL UpdateRect = psbi->rcScrollBar;
               UpdateRect.left -= Window->rcClient.left - Window->rcWindow.left;
