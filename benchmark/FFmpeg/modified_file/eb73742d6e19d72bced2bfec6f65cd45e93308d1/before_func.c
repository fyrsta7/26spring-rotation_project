static void update(Real288_internal *glob)
{
    int x,y;
    float buffer1[40], temp1[37];
    float buffer2[8], temp2[11];

    y = glob->phasep+5;
    for (x=0;  x < 40; x++)
        buffer1[x] = glob->output[(y++)%40];

    co(36, 40, 35, buffer1, temp1, glob->st1a, glob->st1b, table1);

    if (pred(temp1, glob->st1, 36))
        colmult(glob->pr1, glob->st1, table1a, 36);

    y = glob->phase + 1;
    for (x=0; x < 8; x++)
        buffer2[x] = glob->history[(y++) % 8];

    co(10, 8, 20, buffer2, temp2, glob->st2a, glob->st2b, table2);

    if (pred(temp2, glob->st2, 10))
        colmult(glob->pr2, glob->st2, table2a, 10);
}