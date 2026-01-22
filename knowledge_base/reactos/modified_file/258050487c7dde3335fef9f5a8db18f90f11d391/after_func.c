      Result = TRUE;
    }
  else
    {
      /* Message sent by kernel. Convert back to Ansi */
      if (! MsgiAnsiToUnicodeReply(&UcMsg, &AnsiMsg, &Result))
        {
            SPY_ExitMessage(SPY_RESULT_OK, hWnd, Msg, Result, wParam, lParam);
            return FALSE;
        }
    }

  SPY_ExitMessage(SPY_RESULT_OK, hWnd, Msg, Result, wParam, lParam);
  return Result;
}
