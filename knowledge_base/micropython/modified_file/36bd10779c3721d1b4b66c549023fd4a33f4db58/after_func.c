    // If CONTROL.SPSEL is 0, then the exception was stacked up using the
    // main stack pointer (aka MSP). If CONTROL.SPSEL is 1, then the exception
    // was stacked up using the process stack pointer (aka PSP).

    __asm volatile(
    " tst lr, #4    \n"         // Test Bit 3 to see which stack pointer we should use.
    " ite eq        \n"         // Tell the assembler that the nest 2 instructions are if-then-else
    " mrseq r0, msp \n"         // Make R0 point to main stack pointer
    " mrsne r0, psp \n"         // Make R0 point to process stack pointer
    " b HardFault_C_Handler \n" // Off to C land
    );
}
#else
void HardFault_Handler(void) {
    /* Go to infinite loop when Hard Fault exception occurs */
    while (1) {
        __fatal_error("HardFault");
    }
}
#endif // REPORT_HARD_FAULT_REGS

/**
  * @brief   This function handles NMI exception.
  * @param  None
  * @retval None
  */
void NMI_Handler(void) {
}
