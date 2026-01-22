}

// Display a list of entries in the hud, filtered and emphasized based
// on the user input.
R_API char *r_cons_hud(RList *list, const char *prompt, const bool usecolor) {
	const int buf_size = 128;
	int ch, nch, first_line, current_entry_n, j, i = 0;
	char *p, *x, user_input[buf_size], mask[buf_size];
	int last_color_change, top_entry_n = 0;
	char *selected_entry = NULL;
	char tmp, last_mask = 0;
	void *current_entry;
	RListIter *iter;

	user_input[0] = 0;
	r_cons_clear ();
	// Repeat until the user exits the hud
	for (;;) {
		first_line = 1;
		r_cons_gotoxy (0, 0);
		current_entry_n = 0;
		selected_entry = NULL;
		if (prompt && *prompt) {
			r_cons_print (">> ");
			r_cons_println (prompt);
		}
		r_cons_printf ("> %s|\n", user_input);
		int counter = 0;
		int rows = r_cons_get_size (NULL);
		// Iterate over each entry in the list
		r_list_foreach (list, iter, current_entry) {
			memset (mask, 0, buf_size);
			if (!user_input[0] || strmatch (current_entry, user_input, mask, buf_size)) {
				counter++;
				if (counter == rows) {
					break;
				}
				// if the user scrolled down the list, do not print the first entries
				if (!top_entry_n || current_entry_n >= top_entry_n) {
					// remove everything after a tab (in ??, it contains the commands)
					x = strchr (current_entry, '\t');
					if (x) {
						*x = 0;
					}
					p = strdup (current_entry);
					// if the filter is empty, print the entry and move on
					if (!user_input[0]) {
						r_cons_printf (" %c %s\n", first_line? '-': ' ', current_entry);
					} else {
						// otherwise we need to emphasize the matching part
						if (usecolor) {
							last_color_change = 0;
							last_mask = 0;
							r_cons_printf (" %c ", first_line? '-': ' ');
							// Instead of printing one char at the time
							// (which would be slow), we group substrings of the same color
							for (j = 0; p[j] && j < buf_size; j++) {
								if (mask[j] != last_mask) {
									tmp = p[j];
									p[j] = 0;
									if (mask[j]) {
										r_cons_printf (Color_RESET"%s", p + last_color_change);
									} else {
										r_cons_printf (Color_GREEN"%s", p + last_color_change);
									}
									p[j] = tmp;
									last_color_change = j;
									last_mask = mask[j];
								} 
							}
							if (last_mask) {
								r_cons_printf (Color_GREEN"%s\n"Color_RESET, p + last_color_change);
							} else {
								r_cons_printf (Color_RESET"%s\n", p + last_color_change);
							}
						} else {
							// Otherwise we print the matching characters uppercase
							for (j = 0; p[j]; j++) {
								if (mask[j])
									p[j] = toupper ((unsigned char)p[j]);
							}
							r_cons_printf (" %c %s\n", first_line? '-': ' ', p);
						}
					}
					// Clean up and restore the tab character (if any)
					free (p);
					if (x) {
						*x = '\t';
					}
					if (first_line) {
						selected_entry = current_entry;
					}	
					first_line = 0;
				}
				current_entry_n++;
			}
		}

		r_cons_visual_flush ();
		ch = r_cons_readchar ();
		nch = r_cons_arrow_to_hjkl (ch);
		if (nch == 'j' && ch != 'j') {
			if (top_entry_n + 1 < current_entry_n) {
				top_entry_n++;
			}
		} else if (nch == 'k' && ch != 'k') {
			if (top_entry_n >= 0) {
				top_entry_n--;
			}
		} else switch (ch) {
			case 9: // \t
				if (top_entry_n + 1 < current_entry_n) {
					top_entry_n++;
				} else {
					top_entry_n = 0;
				}
				break;
			case 10: // \n
			case 13: // \r
				top_entry_n = 0;
				//		if (!*buf)
				//			return NULL;
				if (current_entry_n >= 1) {
					//eprintf ("%s\n", buf);
					//i = buf[0] = 0;
					return strdup (selected_entry);
				} // no match!
				break;
			case 23: // ^w
				top_entry_n = 0;
				i = user_input[0] = 0;
				break;
			case 0x1b: // ESC
				return NULL;
			case 8:   // bs
			case 127: // bs
				top_entry_n = 0;
				if (i < 1) return NULL;
				user_input[--i] = 0;
				break;
			default:
				if (IS_PRINTABLE (ch)) {
					if (i >= buf_size) {
						break;
					}
					top_entry_n = 0;
					if (i + 1 >= buf_size) {
						// too many
						break;
					}
					user_input[i++] = ch;
					user_input[i] = 0;
				}
				break;
