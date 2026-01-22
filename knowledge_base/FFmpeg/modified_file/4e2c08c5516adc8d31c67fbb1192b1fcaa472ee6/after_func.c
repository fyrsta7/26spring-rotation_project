    evaluate_utility_inc(elbg);

    for (idx[0]=0; idx[0] < elbg->numCB; idx[0]++)
        if (elbg->numCB*elbg->utility[idx[0]] < elbg->error) {
            if (elbg->utility_inc[elbg->numCB-1] == 0)
                return;

            idx[1] = get_high_utility_cell(elbg);
            idx[2] = get_closest_codebook(elbg, idx[0]);

            if (idx[1] != idx[0] && idx[1] != idx[2])
                try_shift_candidate(elbg, idx);
        }
}

#define BIG_PRIME 433494437LL

void ff_init_elbg(int *points, int dim, int numpoints, int *codebook,
                  int numCB, int max_steps, int *closest_cb,
                  AVLFG *rand_state)
{
    int i, k;

    if (numpoints > 24*numCB) {
        /* ELBG is very costly for a big number of points. So if we have a lot
           of them, get a good initial codebook to save on iterations       */
        int *temp_points = av_malloc(dim*(numpoints/8)*sizeof(int));
        for (i=0; i<numpoints/8; i++) {
            k = (i*BIG_PRIME) % numpoints;
            memcpy(temp_points + i*dim, points + k*dim, dim*sizeof(int));
        }

        ff_init_elbg(temp_points, dim, numpoints/8, codebook, numCB, 2*max_steps, closest_cb, rand_state);
        ff_do_elbg(temp_points, dim, numpoints/8, codebook, numCB, 2*max_steps, closest_cb, rand_state);

        av_free(temp_points);

    } else  // If not, initialize the codebook with random positions
        for (i=0; i < numCB; i++)
            memcpy(codebook + i*dim, points + ((i*BIG_PRIME)%numpoints)*dim,
                   dim*sizeof(int));

}

void ff_do_elbg(int *points, int dim, int numpoints, int *codebook,
                int numCB, int max_steps, int *closest_cb,
                AVLFG *rand_state)
{
    int dist;
    elbg_data elbg_d;
    elbg_data *elbg = &elbg_d;
    int i, j, k, last_error, steps=0;
    int *dist_cb = av_malloc(numpoints*sizeof(int));
    int *size_part = av_malloc(numCB*sizeof(int));
    cell *list_buffer = av_malloc(numpoints*sizeof(cell));
    cell *free_cells;
    int best_dist, best_idx = 0;

    elbg->error = INT_MAX;
    elbg->dim = dim;
    elbg->numCB = numCB;
    elbg->codebook = codebook;
    elbg->cells = av_malloc(numCB*sizeof(cell *));
    elbg->utility = av_malloc(numCB*sizeof(int));
    elbg->nearest_cb = closest_cb;
    elbg->points = points;
    elbg->utility_inc = av_malloc(numCB*sizeof(int));

    elbg->rand_state = rand_state;

    do {
        free_cells = list_buffer;
        last_error = elbg->error;
        steps++;
        memset(elbg->utility, 0, numCB*sizeof(int));
        memset(elbg->cells, 0, numCB*sizeof(cell *));

        elbg->error = 0;

        /* This loop evaluate the actual Voronoi partition. It is the most
           costly part of the algorithm. */
        for (i=0; i < numpoints; i++) {
