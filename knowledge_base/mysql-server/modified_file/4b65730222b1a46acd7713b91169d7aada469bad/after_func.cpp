  return
    globalData.testOn &&
    globalSignalLoggers.logMatch(number(), mask);
}

const char*
SimulatedBlock::debugOutTag(char *buf, int line)
{
  char blockbuf[40];
  char instancebuf[40];
  char linebuf[40];
  char timebuf[40];
  sprintf(blockbuf, "%s", getBlockName(number(), "UNKNOWN"));
  instancebuf[0] = 0;
  if (instance() != 0)
    sprintf(instancebuf, "/%u", instance());
  sprintf(linebuf, " %d", line);
  timebuf[0] = 0;
#ifdef VM_TRACE_TIME
  {
    Uint64 t = NdbTick_CurrentMillisecond();
    uint s = (t / 1000) % 3600;
    uint ms = t % 1000;
    sprintf(timebuf, " - %u.%03u -", s, ms);
  }
#endif
  sprintf(buf, "%s%s%s%s ", blockbuf, instancebuf, linebuf, timebuf);
  return buf;
}
#endif

void
SimulatedBlock::synchronize_threads_for_blocks(Signal * signal,
                                               const Uint32 blocks[],
                                               const Callback & cb,
                                               JobBufferLevel prio)
{
#ifndef NDBD_MULTITHREADED
  Callback copy = cb;
  execute(signal, copy, 0);
#else
  jam();
  Uint32 ref[32]; // max threads
