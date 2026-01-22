    {NULL,           NULL,             IDS_NONE,                 IDS_NONE}
};


/* FUNCTIONS *****************************************************************/

/*
 * InterpretCmd(char *cmd_line, char *arg_line):
 * compares the command name to a list of available commands, and
 * determines which function to envoke.
 */
BOOL
InterpretCmd(int argc, LPWSTR *argv)
{
    PCOMMAND cmdptr;

    /* Scan internal command table */
    for (cmdptr = cmds; cmdptr->name; cmdptr++)
    {
        /* First, determine if the user wants to exit
        or to use a comment */
        if(wcsicmp(argv[0], L"exit") == 0)
            return FALSE;
