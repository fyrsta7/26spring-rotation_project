
    // if we have processed all the characters, reset the buffer counters
    if (UserRxBufCur > 0 && UserRxBufCur >= UserRxBufLen) {
        memmove(UserRxBuffer, UserRxBuffer + UserRxBufLen, *Len);
        UserRxBufCur = 0;
        UserRxBufLen = 0;
    }

    uint32_t delta_len;

    if (user_interrupt_char == VCP_CHAR_NONE) {
        // no special interrupt character
        delta_len = *Len;
