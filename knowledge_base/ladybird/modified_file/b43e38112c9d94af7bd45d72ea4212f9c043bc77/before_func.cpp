    // end:
    end.link(m_assembler);
}

void Compiler::compile_jump_conditional(Bytecode::Op::JumpConditional const& op)
{
    load_vm_register(GPR1, Bytecode::Register::accumulator());

    compile_to_boolean(GPR0, GPR1);

    m_assembler.jump_if_equal(
        Assembler::Operand::Register(GPR0),
        Assembler::Operand::Imm32(0),
