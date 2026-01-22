        {
            /* Mutually exclusive with REPNZ */
            if (!(State->PrefixFlags
                & (FAST486_PREFIX_REPNZ | FAST486_PREFIX_REP)))
            {
                State->PrefixFlags |= FAST486_PREFIX_REP;
                Valid = TRUE;
            }

            break;
        }
    }

    if (!Valid)
    {
        /* Clear all prefixes */
        State->PrefixFlags = 0;

        /* Throw an exception */
        Fast486Exception(State, FAST486_EXCEPTION_UD);
    }
}

FAST486_OPCODE_HANDLER(Fast486OpcodeIncrement)
{
    ULONG Value;
    BOOLEAN Size = State->SegmentRegs[FAST486_REG_CS].Size;

    TOGGLE_OPSIZE(Size);
    NO_LOCK_PREFIX();
