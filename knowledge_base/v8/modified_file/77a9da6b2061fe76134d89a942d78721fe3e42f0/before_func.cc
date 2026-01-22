void ArrayNArgumentsConstructorStub::Generate(MacroAssembler* masm) {
  __ pop(ecx);
  __ mov(MemOperand(esp, eax, times_4, 0), edi);
  __ push(edi);
  __ push(ebx);
  __ push(ecx);
  __ add(eax, Immediate(3));
  __ TailCallRuntime(Runtime::kNewArray);
}
