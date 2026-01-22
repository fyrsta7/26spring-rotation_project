                ret2 += ret;
            }
            if(in_count){
                buf_set(&tmp, &s->in_buffer, s->in_buffer_index + s->in_buffer_count);
                copy(&tmp, in, in_count);
                s->in_buffer_count += in_count;
            }
        }
        if(ret2>0 && !s->drop_output)
            s->outpts += ret2 * (int64_t)s->in_sample_rate;
        return ret2;
    }
}

int swr_drop_output(struct SwrContext *s, int count){
    s->drop_output += count;

    if(s->drop_output <= 0)
        return 0;

