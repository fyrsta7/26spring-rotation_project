{
    queue_size_ms = (int)(queue_size * 1000);
}

int SrsMessageQueue::enqueue(SrsSharedPtrMessage* msg, bool* is_overflow)
{
    int ret = ERROR_SUCCESS;
    
    if (msg->is_av()) {
        if (av_start_time == -1) {
            av_start_time = msg->timestamp;
        }
        
        av_end_time = msg->timestamp;
    }
    
    msgs.push_back(msg);

    while (av_end_time - av_start_time > queue_size_ms) {
        // notice the caller queue already overflow and shrinked.
        if (is_overflow) {
            *is_overflow = true;
        }
        
        shrink();
    }
    
    return ret;
}

int SrsMessageQueue::dump_packets(int max_count, SrsSharedPtrMessage** pmsgs, int& count)
{
    int ret = ERROR_SUCCESS;
