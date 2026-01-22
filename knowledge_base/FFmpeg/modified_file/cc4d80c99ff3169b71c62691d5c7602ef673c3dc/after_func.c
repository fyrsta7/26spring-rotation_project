            s->decode_map_chunk_size = opcode_size;
            avio_skip(pb, opcode_size);
            break;

        case OPCODE_VIDEO_DATA:
            av_dlog(NULL, "set video data\n");

            /* log position and move on for now */
            s->video_chunk_offset = avio_tell(pb);
            s->video_chunk_size = opcode_size;
            avio_skip(pb, opcode_size);
            break;
