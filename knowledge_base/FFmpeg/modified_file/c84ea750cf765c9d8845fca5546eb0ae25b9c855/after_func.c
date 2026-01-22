
static void add_pid_to_pmt(MpegTSContext *ts, unsigned int programid, unsigned int pid)
{
    int i;
    struct Program *p = NULL;
    for(i=0; i<ts->nb_prg; i++) {
        if(ts->prg[i].id == programid) {
            p = &ts->prg[i];
            break;
        }
    }
    if(!p)
        return;

    if(p->nb_pids >= MAX_PIDS_PER_PROGRAM)
        return;
    p->pids[p->nb_pids++] = pid;
}

/**
 * @brief discard_pid() decides if the pid is to be discarded according
 *                      to caller's programs selection
 * @param ts    : - TS context
 * @param pid   : - pid
 * @return 1 if the pid is only comprised in programs that have .discard=AVDISCARD_ALL
 *         0 otherwise
 */
static int discard_pid(MpegTSContext *ts, unsigned int pid)
{
    int i, j, k;
    int used = 0, discarded = 0;
    struct Program *p;

