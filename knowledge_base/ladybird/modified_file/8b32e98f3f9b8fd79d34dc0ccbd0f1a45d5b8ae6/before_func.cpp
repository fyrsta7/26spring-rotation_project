
    not_int32_case.link(m_assembler);
}

void Compiler::compile_increment(Bytecode::Op::Increment const&)
{
    load_vm_register(ARG1, Bytecode::Register::accumulator());

    auto end = m_assembler.make_label();
    auto slow_case = m_assembler.make_label();

    branch_if_int32(ARG1, [&] {
        // GPR0 = ARG1 & 0xffffffff;
        m_assembler.mov(
            Assembler::Operand::Register(GPR0),
            Assembler::Operand::Register(ARG1));
        m_assembler.mov(
            Assembler::Operand::Register(GPR1),
            Assembler::Operand::Imm64(0xffffffff));
        m_assembler.bitwise_and(
            Assembler::Operand::Register(GPR0),
            Assembler::Operand::Register(GPR1));

        // if (GPR0 == 0x7fffffff) goto slow_case;
        m_assembler.jump_if_equal(
            Assembler::Operand::Register(GPR0),
            Assembler::Operand::Imm32(0x7fffffff),
            slow_case);

        // GPR0 += 1;
        m_assembler.add(
            Assembler::Operand::Register(GPR0),
            Assembler::Operand::Imm32(1));

        // GPR0 |= INT32_TAG << 48;
        m_assembler.mov(
            Assembler::Operand::Register(GPR1),
            Assembler::Operand::Imm64(SHIFTED_INT32_TAG));
        m_assembler.bitwise_or(
            Assembler::Operand::Register(GPR0),
            Assembler::Operand::Register(GPR1));

        // accumulator = GPR0;
        store_vm_register(Bytecode::Register::accumulator(), GPR0);

        m_assembler.jump(end);
    });

    slow_case.link(m_assembler);
    m_assembler.native_call((void*)cxx_increment);
    store_vm_register(Bytecode::Register::accumulator(), RET);
