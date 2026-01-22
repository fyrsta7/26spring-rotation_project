NTAPI
SepInitPrivileges(VOID)
{

}


BOOLEAN
NTAPI
SepPrivilegeCheck(PTOKEN Token,
                  PLUID_AND_ATTRIBUTES Privileges,
                  ULONG PrivilegeCount,
                  ULONG PrivilegeControl,
                  KPROCESSOR_MODE PreviousMode)
{
    ULONG i;
    ULONG j;
    ULONG Required;

    DPRINT("SepPrivilegeCheck() called\n");

    PAGED_CODE();

    if (PreviousMode == KernelMode)
        return TRUE;

    /* Get the number of privileges that are required to match */
    Required = (PrivilegeControl & PRIVILEGE_SET_ALL_NECESSARY) ? PrivilegeCount : 1;

    /* Loop all requested privileges until we found the required ones */
    for (i = 0; i < PrivilegeCount; i++)
    {
        /* Loop the privileges of the token */
        for (j = 0; j < Token->PrivilegeCount; j++)
        {
            /* Check if the LUIDs match */
            if (Token->Privileges[j].Luid.LowPart == Privileges[i].Luid.LowPart &&
                Token->Privileges[j].Luid.HighPart == Privileges[i].Luid.HighPart)
            {
                DPRINT("Found privilege. Attributes: %lx\n",
                       Token->Privileges[j].Attributes);

                /* Check if the privilege is enabled */
                if (Token->Privileges[j].Attributes & SE_PRIVILEGE_ENABLED)
                {
                    Privileges[i].Attributes |= SE_PRIVILEGE_USED_FOR_ACCESS;
                    Required--;

                    /* Check if we have found all privileges */
                    if (Required == 0)
                    {
                        /* We're done! */
                        return TRUE;
                    }
                }

                /* Leave the inner loop */
                break;
            }
