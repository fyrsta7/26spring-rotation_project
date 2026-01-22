				/* Truncate left if still too long. */
				while (scridx[iend] - scridx[ibeg] > DISPLAY_SIZE)
				{
					ibeg++;
					beg_trunc = true;
				}
			}
		}

		/* truncate working copy at desired endpoint */
		wquery[qidx[iend]] = '\0';

		/* Begin building the finished message. */
		i = msg->len;
		appendPQExpBuffer(msg, libpq_gettext("LINE %d: "), loc_line);
		if (beg_trunc)
			appendPQExpBufferStr(msg, "...");

		/*
		 * While we have the prefix in the msg buffer, compute its screen
		 * width.
		 */
		scroffset = 0;
		for (; i < msg->len; i += pg_encoding_mblen(encoding, &msg->data[i]))
		{
			int			w = pg_encoding_dsplen(encoding, &msg->data[i]);

			if (w <= 0)
				w = 1;
			scroffset += w;
		}

		/* Finish up the LINE message line. */
		appendPQExpBufferStr(msg, &wquery[qidx[ibeg]]);
		if (end_trunc)
			appendPQExpBufferStr(msg, "...");
		appendPQExpBufferChar(msg, '\n');

		/* Now emit the cursor marker line. */
		scroffset += scridx[loc] - scridx[ibeg];
		for (i = 0; i < scroffset; i++)
			appendPQExpBufferChar(msg, ' ');
		appendPQExpBufferChar(msg, '^');
		appendPQExpBufferChar(msg, '\n');
	}

	/* Clean up. */
	free(scridx);
	free(qidx);
	free(wquery);
}


/*
 * Attempt to read a ParameterStatus message.
 * This is possible in several places, so we break it out as a subroutine.
 * Entry: 'S' message type and length have already been consumed.
 * Exit: returns 0 if successfully consumed message.
