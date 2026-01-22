#include <debug.h>

#include <fast486.h>
#include "common.h"
#include "opcodes.h"
#include "fpu.h"

/* DEFINES ********************************************************************/

typedef enum
{
    FAST486_STEP_INTO,
    FAST486_STEP_OVER,
    FAST486_STEP_OUT,
    FAST486_CONTINUE
} FAST486_EXEC_CMD;

/* PRIVATE FUNCTIONS **********************************************************/

static inline VOID
NTAPI
Fast486ExecutionControl(PFAST486_STATE State, FAST486_EXEC_CMD Command)
{
    UCHAR Opcode;
    FAST486_OPCODE_HANDLER_PROC CurrentHandler;
    INT ProcedureCallCount = 0;

    /* Main execution loop */
    do
    {
NextInst:
        /* Check if this is a new instruction */
        if (State->PrefixFlags == 0) State->SavedInstPtr = State->InstPtr;

        /* Perform an instruction fetch */
        if (!Fast486FetchByte(State, &Opcode))
        {
            /* Exception occurred */
            State->PrefixFlags = 0;
            continue;
        }

        // TODO: Check for CALL/RET to update ProcedureCallCount.

        /* Call the opcode handler */
        CurrentHandler = Fast486OpcodeHandlers[Opcode];
        CurrentHandler(State, Opcode);

        /* If this is a prefix, go to the next instruction immediately */
        if (CurrentHandler == Fast486OpcodePrefix) goto NextInst;

        /* A non-prefix opcode has been executed, reset the prefix flags */
        State->PrefixFlags = 0;

        /*
         * Check if there is an interrupt to execute, or a hardware interrupt signal
         * while interrupts are enabled.
         */
        if (State->IntStatus == FAST486_INT_EXECUTE)
        {
            /* Perform the interrupt */
            Fast486PerformInterrupt(State, State->PendingIntNum);

