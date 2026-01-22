
  delay_count >>= 1;              /* Get bottom value for delay     */

  /* Stage 2:  Fine calibration                                     */
  DbgPrint("delay_count: %d", delay_count);

  calib_bit = delay_count;        /* Which bit are we going to test */

  for (i = 0; i < PRECISION; i++)
    {
      calib_bit >>= 1;            /* Next bit to calibrate          */
      if (!calib_bit)
	{
	  break;                  /* If we have done all bits, stop */
