}


void* kmalloc(dword size)
{
    InterruptDisabler disabler;

    DWORD chunks_needed, chunks_here, first_chunk;
    DWORD real_size;
    DWORD i, j, k;

    /* We need space for the allocation_t structure at the head of the block. */
    real_size = size + sizeof(allocation_t);

    if (sum_free < real_size) {
        kprintf("kmalloc(): PANIC! Out of memory (sucks, dude)\nsum_free=%u, real_size=%x\n", sum_free, real_size);
        HANG;
        return 0L;
    }

    chunks_needed = real_size / CHUNK_SIZE;
    if( real_size % CHUNK_SIZE )
        chunks_needed++;

    chunks_here = 0;
    first_chunk = 0;

    for( i = 0; i < (POOL_SIZE / CHUNK_SIZE / 8); ++i )
    {
        if (alloc_map[i] == 0xff) {
            // Skip over completely full bucket.
            chunks_here = 0;
            continue;
        }
        // FIXME: This scan can be optimized further with LZCNT.
        for( j = 0; j < 8; ++j )
        {
            if( !(alloc_map[i] & (1<<j)) )
            {
                if( chunks_here == 0 )
                {
                    /* Mark where potential allocation starts. */
                    first_chunk = i * 8 + j;
                }

                chunks_here++;

                if( chunks_here == chunks_needed )
                {
                    auto* a = (allocation_t *)(BASE_PHYS + (first_chunk * CHUNK_SIZE));
                    BYTE *ptr = (BYTE *)a;
                    ptr += sizeof(allocation_t);
                    a->nchunk = chunks_needed;
                    a->start = first_chunk;

                    for( k = first_chunk; k < (first_chunk + chunks_needed); ++k )
                    {
                        alloc_map[k / 8] |= 1 << (k % 8);
                    }

                    sum_alloc += a->nchunk * CHUNK_SIZE;
                    sum_free  -= a->nchunk * CHUNK_SIZE;
#ifdef SANITIZE_KMALLOC
                    memset(ptr, 0xbb, (a->nchunk * CHUNK_SIZE) - sizeof(allocation_t));
#endif
                    return ptr;
                }
            }
            else
            {
                /* This is in use, so restart chunks_here counter. */
                chunks_here = 0;
            }
        }
    }

    kprintf("kmalloc(): PANIC! Out of memory (no suitable block for size %u)\n", size);
    HANG;
