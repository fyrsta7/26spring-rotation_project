{
    TRY_OR_SET_EXCEPTION(Bytecode::put_by_value(vm, base, property, value, kind));
    return value;
}

void Compiler::compile_put_by_value(Bytecode::Op::PutByValue const& op)
{
    load_vm_register(ARG1, op.base());
    load_vm_register(ARG2, op.property());

    Assembler::Label end {};
    Assembler::Label slow_case {};

    branch_if_object(ARG1, [&] {
        branch_if_int32(ARG2, [&] {
            // if (ARG2 < 0) goto slow_case;
            m_assembler.mov(
                Assembler::Operand::Register(GPR0),
                Assembler::Operand::Register(ARG2));
            m_assembler.sign_extend_32_to_64_bits(GPR0);
            m_assembler.jump_if(
                Assembler::Operand::Register(GPR0),
                Assembler::Condition::SignedLessThan,
                Assembler::Operand::Imm(0),
                slow_case);

            // GPR0 = extract_pointer(ARG1)
            extract_object_pointer(GPR0, ARG1);

            // if (object->may_interfere_with_indexed_property_access()) goto slow_case;
            m_assembler.mov8(
                Assembler::Operand::Register(GPR1),
                Assembler::Operand::Mem64BaseAndOffset(GPR0, Object::may_interfere_with_indexed_property_access_offset()));
            m_assembler.jump_if(
                Assembler::Operand::Register(GPR1),
                Assembler::Condition::NotEqualTo,
                Assembler::Operand::Imm(0),
                slow_case);

            // GPR0 = object->indexed_properties().storage()
            m_assembler.mov(
                Assembler::Operand::Register(GPR0),
                Assembler::Operand::Mem64BaseAndOffset(GPR0, Object::indexed_properties_offset() + IndexedProperties::storage_offset()));

            // if (GPR0 == nullptr) goto slow_case;
            m_assembler.jump_if(
                Assembler::Operand::Register(GPR0),
                Assembler::Condition::EqualTo,
                Assembler::Operand::Imm(0),
                slow_case);

            // if (!GPR0->is_simple_storage()) goto slow_case;
            m_assembler.mov8(
                Assembler::Operand::Register(GPR1),
                Assembler::Operand::Mem64BaseAndOffset(GPR0, IndexedPropertyStorage::is_simple_storage_offset()));
            m_assembler.jump_if(
                Assembler::Operand::Register(GPR1),
                Assembler::Condition::EqualTo,
                Assembler::Operand::Imm(0),
                slow_case);

            // GPR2 = extract_int32(ARG2)
            m_assembler.mov32(
                Assembler::Operand::Register(GPR2),
                Assembler::Operand::Register(ARG2));

            // if (GPR2 >= GPR0->array_like_size()) goto slow_case;
            m_assembler.mov(
                Assembler::Operand::Register(GPR1),
                Assembler::Operand::Mem64BaseAndOffset(GPR0, SimpleIndexedPropertyStorage::array_size_offset()));
            m_assembler.jump_if(
                Assembler::Operand::Register(GPR2),
                Assembler::Condition::SignedGreaterThanOrEqualTo,
                Assembler::Operand::Register(GPR1),
                slow_case);

            // GPR0 = GPR0->elements().outline_buffer()
            m_assembler.mov(
                Assembler::Operand::Register(GPR0),
                Assembler::Operand::Mem64BaseAndOffset(GPR0, SimpleIndexedPropertyStorage::elements_offset() + Vector<Value>::outline_buffer_offset()));

            // GPR2 *= sizeof(Value)
            m_assembler.mul32(
                Assembler::Operand::Register(GPR2),
                Assembler::Operand::Imm(sizeof(Value)),
                slow_case);

            // GPR0 = &GRP0[GPR2]
            // GPR2 = *GPR0
            m_assembler.add(
                Assembler::Operand::Register(GPR0),
                Assembler::Operand::Register(GPR2));
            m_assembler.mov(
                Assembler::Operand::Register(GPR2),
                Assembler::Operand::Mem64BaseAndOffset(GPR0, 0));

            // if (GPR2.is_accessor()) goto slow_case;
            m_assembler.mov(Assembler::Operand::Register(GPR1), Assembler::Operand::Register(GPR2));
            m_assembler.shift_right(Assembler::Operand::Register(GPR1), Assembler::Operand::Imm(TAG_SHIFT));
            m_assembler.jump_if(
                Assembler::Operand::Register(GPR1),
                Assembler::Condition::EqualTo,
                Assembler::Operand::Imm(ACCESSOR_TAG),
                slow_case);

            // GRP1 will clobber ARG3 in X86, so load it later.
            load_accumulator(ARG3);

            // *GPR0 = value
            m_assembler.mov(
                Assembler::Operand::Mem64BaseAndOffset(GPR0, 0),
                Assembler::Operand::Register(ARG3));

            // accumulator = ARG3;
            store_accumulator(ARG3);
            m_assembler.jump(end);
        });
    });

    slow_case.link(m_assembler);
    load_accumulator(ARG3);
    m_assembler.mov(
        Assembler::Operand::Register(ARG4),
        Assembler::Operand::Imm(to_underlying(op.kind())));
