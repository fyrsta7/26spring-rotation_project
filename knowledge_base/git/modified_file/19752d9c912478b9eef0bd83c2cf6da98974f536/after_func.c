	 * This count excludes parents that won't be printed in the graph
	 * output, as determined by graph_is_interesting().
	 */
	int num_parents;
	/*
	 * The width of the graph output for this commit.
	 * All rows for this commit are padded to this width, so that
	 * messages printed after the graph output are aligned.
	 */
	int width;
	/*
	 * The next expansion row to print
	 * when state is GRAPH_PRE_COMMIT
	 */
	int expansion_row;
	/*
	 * The current output state.
