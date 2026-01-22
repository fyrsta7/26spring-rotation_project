
    u32 activeIdx = lastActiveIdx;
    // If we have top events in the main queue, update current active id
    if (q1->cur < q1->end - 1) {
        const u32 *baseTop = (const u32 *)((const char *)t +
                                           sizeof(struct Tamarama));
        u32 curTop = q1->items[q1->cur].type;
        activeIdx = findEngineForTop(baseTop, curTop, numSubEngines);
    }

    assert(activeIdx < numSubEngines);
    DEBUG_PRINTF("last id:%u, current id:%u, num of subengines:%u\n",
                 lastActiveIdx, activeIdx, numSubEngines);
    // Handle unfinished last alive subengine
    if (lastActiveIdx != activeIdx &&
        lastActiveIdx != numSubEngines && hasStart) {
        loc = q1->items[q1->cur].location;
        pushQueueNoMerge(q2, MQE_END, loc);
        q2->nfa = getSubEngine(t, lastActiveIdx);
        return;
    }

    initSubQueue(t, q1, q2, lastActiveIdx, activeIdx);
    DEBUG_PRINTF("finish queues\n");
}

// After processing subqueue items for subengines, we need to copy back
// remaining items in subqueue if there are any to Tamarama main queue
static
void copyBack(const struct  Tamarama *t, struct mq *q, struct mq *q1) {
    DEBUG_PRINTF("copy back %u, %u\n", q1->cur, q1->end);
    q->report_current = q1->report_current;
    if (q->cur >= q->end && q1->cur >= q1->end) {
        return;
    }

    const u32 *baseTop = (const u32 *)((const char *)t +
                                        sizeof(struct Tamarama));
    const u32 lastIdx = loadActiveIdx(q->streamState,
                                      t->activeIdxSize);
    u32 base = 0, event_base = 0;
    if (lastIdx != t->numSubEngines) {
        base = baseTop[lastIdx];
