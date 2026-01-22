CodeGenerator::CodeGenResult CodeGenerator::AssembleArchInstruction(
    Instruction* instr) {
  X64OperandConverter i(this, instr);
  InstructionCode opcode = instr->opcode();
  ArchOpcode arch_opcode = ArchOpcodeField::decode(opcode);
  switch (arch_opcode) {
    case kArchCallCodeObject: {
      if (HasImmediateInput(instr, 0)) {
        Handle<Code> code = i.InputCode(0);
        __ Call(code, RelocInfo::CODE_TARGET);
      } else {
        Register reg = i.InputRegister(0);
        DCHECK_IMPLIES(
            instr->HasCallDescriptorFlag(CallDescriptor::kFixedTargetRegister),
            reg == kJavaScriptCallCodeStartRegister);
        __ LoadCodeObjectEntry(reg, reg);
        if (instr->HasCallDescriptorFlag(CallDescriptor::kRetpoline)) {
          __ RetpolineCall(reg);
        } else {
          __ call(reg);
        }
      }
      RecordCallPosition(instr);
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchCallBuiltinPointer: {
      DCHECK(!HasImmediateInput(instr, 0));
      Register builtin_index = i.InputRegister(0);
      __ CallBuiltinByIndex(builtin_index);
      RecordCallPosition(instr);
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchCallWasmFunction: {
      if (HasImmediateInput(instr, 0)) {
        Constant constant = i.ToConstant(instr->InputAt(0));
        Address wasm_code = static_cast<Address>(constant.ToInt64());
        if (DetermineStubCallMode() == StubCallMode::kCallWasmRuntimeStub) {
          __ near_call(wasm_code, constant.rmode());
        } else {
          if (instr->HasCallDescriptorFlag(CallDescriptor::kRetpoline)) {
            __ RetpolineCall(wasm_code, constant.rmode());
          } else {
            __ Call(wasm_code, constant.rmode());
          }
        }
      } else {
        Register reg = i.InputRegister(0);
        if (instr->HasCallDescriptorFlag(CallDescriptor::kRetpoline)) {
          __ RetpolineCall(reg);
        } else {
          __ call(reg);
        }
      }
      RecordCallPosition(instr);
      frame_access_state()->ClearSPDelta();
      break;
    }
    case kArchTailCallCodeObjectFromJSFunction:
      if (!instr->HasCallDescriptorFlag(CallDescriptor::kIsTailCallForTierUp)) {
        AssemblePopArgumentsAdaptorFrame(kJavaScriptCallArgCountRegister,
                                         i.TempRegister(0), i.TempRegister(1),
                                         i.TempRegister(2));
      }
      V8_FALLTHROUGH;
    case kArchTailCallCodeObject: {
      if (HasImmediateInput(instr, 0)) {
        Handle<Code> code = i.InputCode(0);
        __ Jump(code, RelocInfo::CODE_TARGET);
      } else {
        Register reg = i.InputRegister(0);
        DCHECK_IMPLIES(
            instr->HasCallDescriptorFlag(CallDescriptor::kFixedTargetRegister),
            reg == kJavaScriptCallCodeStartRegister);
        __ LoadCodeObjectEntry(reg, reg);
        if (instr->HasCallDescriptorFlag(CallDescriptor::kRetpoline)) {
          __ RetpolineJump(reg);
        } else {
          __ jmp(reg);
        }
      }
      unwinding_info_writer_.MarkBlockWillExit();
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchTailCallWasm: {
      if (HasImmediateInput(instr, 0)) {
        Constant constant = i.ToConstant(instr->InputAt(0));
        Address wasm_code = static_cast<Address>(constant.ToInt64());
        if (DetermineStubCallMode() == StubCallMode::kCallWasmRuntimeStub) {
          __ near_jmp(wasm_code, constant.rmode());
        } else {
          __ Move(kScratchRegister, wasm_code, constant.rmode());
          __ jmp(kScratchRegister);
        }
      } else {
        Register reg = i.InputRegister(0);
        if (instr->HasCallDescriptorFlag(CallDescriptor::kRetpoline)) {
          __ RetpolineJump(reg);
        } else {
          __ jmp(reg);
        }
      }
      unwinding_info_writer_.MarkBlockWillExit();
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchTailCallAddress: {
      CHECK(!HasImmediateInput(instr, 0));
      Register reg = i.InputRegister(0);
      DCHECK_IMPLIES(
          instr->HasCallDescriptorFlag(CallDescriptor::kFixedTargetRegister),
          reg == kJavaScriptCallCodeStartRegister);
      if (instr->HasCallDescriptorFlag(CallDescriptor::kRetpoline)) {
        __ RetpolineJump(reg);
      } else {
        __ jmp(reg);
      }
      unwinding_info_writer_.MarkBlockWillExit();
      frame_access_state()->ClearSPDelta();
      frame_access_state()->SetFrameAccessToDefault();
      break;
    }
    case kArchCallJSFunction: {
      Register func = i.InputRegister(0);
      if (FLAG_debug_code) {
        // Check the function's context matches the context argument.
        __ cmp_tagged(rsi, FieldOperand(func, JSFunction::kContextOffset));
        __ Assert(equal, AbortReason::kWrongFunctionContext);
      }
      static_assert(kJavaScriptCallCodeStartRegister == rcx, "ABI mismatch");
      __ LoadTaggedPointerField(rcx,
                                FieldOperand(func, JSFunction::kCodeOffset));
      __ CallCodeObject(rcx);
      frame_access_state()->ClearSPDelta();
      RecordCallPosition(instr);
      break;
    }
    case kArchPrepareCallCFunction: {
      // Frame alignment requires using FP-relative frame addressing.
      frame_access_state()->SetFrameAccessToFP();
      int const num_parameters = MiscField::decode(instr->opcode());
      __ PrepareCallCFunction(num_parameters);
      break;
    }
    case kArchSaveCallerRegisters: {
      fp_mode_ =
          static_cast<SaveFPRegsMode>(MiscField::decode(instr->opcode()));
      DCHECK(fp_mode_ == kDontSaveFPRegs || fp_mode_ == kSaveFPRegs);
      // kReturnRegister0 should have been saved before entering the stub.
      int bytes = __ PushCallerSaved(fp_mode_, kReturnRegister0);
      DCHECK(IsAligned(bytes, kSystemPointerSize));
      DCHECK_EQ(0, frame_access_state()->sp_delta());
      frame_access_state()->IncreaseSPDelta(bytes / kSystemPointerSize);
      DCHECK(!caller_registers_saved_);
      caller_registers_saved_ = true;
      break;
    }
    case kArchRestoreCallerRegisters: {
      DCHECK(fp_mode_ ==
             static_cast<SaveFPRegsMode>(MiscField::decode(instr->opcode())));
      DCHECK(fp_mode_ == kDontSaveFPRegs || fp_mode_ == kSaveFPRegs);
      // Don't overwrite the returned value.
      int bytes = __ PopCallerSaved(fp_mode_, kReturnRegister0);
      frame_access_state()->IncreaseSPDelta(-(bytes / kSystemPointerSize));
      DCHECK_EQ(0, frame_access_state()->sp_delta());
      DCHECK(caller_registers_saved_);
      caller_registers_saved_ = false;
      break;
    }
    case kArchPrepareTailCall:
      AssemblePrepareTailCall();
      break;
    case kArchCallCFunction: {
      int const num_parameters = MiscField::decode(instr->opcode());
      Label return_location;
      if (linkage()->GetIncomingDescriptor()->IsWasmCapiFunction()) {
        // Put the return address in a stack slot.
        __ leaq(kScratchRegister, Operand(&return_location, 0));
        __ movq(MemOperand(rbp, WasmExitFrameConstants::kCallingPCOffset),
                kScratchRegister);
      }
      if (HasImmediateInput(instr, 0)) {
        ExternalReference ref = i.InputExternalReference(0);
        __ CallCFunction(ref, num_parameters);
      } else {
        Register func = i.InputRegister(0);
        __ CallCFunction(func, num_parameters);
      }
      __ bind(&return_location);
      if (linkage()->GetIncomingDescriptor()->IsWasmCapiFunction()) {
        RecordSafepoint(instr->reference_map(), Safepoint::kNoLazyDeopt);
      }
      frame_access_state()->SetFrameAccessToDefault();
      // Ideally, we should decrement SP delta to match the change of stack
      // pointer in CallCFunction. However, for certain architectures (e.g.
      // ARM), there may be more strict alignment requirement, causing old SP
      // to be saved on the stack. In those cases, we can not calculate the SP
      // delta statically.
      frame_access_state()->ClearSPDelta();
      if (caller_registers_saved_) {
        // Need to re-sync SP delta introduced in kArchSaveCallerRegisters.
        // Here, we assume the sequence to be:
        //   kArchSaveCallerRegisters;
        //   kArchCallCFunction;
        //   kArchRestoreCallerRegisters;
        int bytes =
            __ RequiredStackSizeForCallerSaved(fp_mode_, kReturnRegister0);
        frame_access_state()->IncreaseSPDelta(bytes / kSystemPointerSize);
      }
      // TODO(tebbi): Do we need an lfence here?
      break;
    }
    case kArchJmp:
      AssembleArchJump(i.InputRpo(0));
      break;
    case kArchBinarySearchSwitch:
      AssembleArchBinarySearchSwitch(instr);
      break;
    case kArchTableSwitch:
      AssembleArchTableSwitch(instr);
      break;
    case kArchComment:
      __ RecordComment(reinterpret_cast<const char*>(i.InputInt64(0)));
      break;
    case kArchAbortCSAAssert:
      DCHECK(i.InputRegister(0) == rdx);
      {
        // We don't actually want to generate a pile of code for this, so just
        // claim there is a stack frame, without generating one.
        FrameScope scope(tasm(), StackFrame::NONE);
        __ Call(
            isolate()->builtins()->builtin_handle(Builtins::kAbortCSAAssert),
            RelocInfo::CODE_TARGET);
      }
      __ int3();
      unwinding_info_writer_.MarkBlockWillExit();
      break;
    case kArchDebugBreak:
      __ DebugBreak();
      break;
    case kArchThrowTerminator:
      unwinding_info_writer_.MarkBlockWillExit();
      break;
    case kArchNop:
      // don't emit code for nops.
      break;
    case kArchDeoptimize: {
      DeoptimizationExit* exit =
          BuildTranslation(instr, -1, 0, OutputFrameStateCombine::Ignore());
      __ jmp(exit->label());
      break;
    }
    case kArchRet:
      AssembleReturn(instr->InputAt(0));
      break;
    case kArchFramePointer:
      __ movq(i.OutputRegister(), rbp);
      break;
    case kArchParentFramePointer:
      if (frame_access_state()->has_frame()) {
        __ movq(i.OutputRegister(), Operand(rbp, 0));
      } else {
        __ movq(i.OutputRegister(), rbp);
      }
      break;
    case kArchStackPointerGreaterThan: {
      // Potentially apply an offset to the current stack pointer before the
      // comparison to consider the size difference of an optimized frame versus
      // the contained unoptimized frames.

      Register lhs_register = rsp;
      uint32_t offset;

      if (ShouldApplyOffsetToStackCheck(instr, &offset)) {
        lhs_register = kScratchRegister;
        __ leaq(lhs_register, Operand(rsp, static_cast<int32_t>(offset) * -1));
      }

      constexpr size_t kValueIndex = 0;
      if (HasAddressingMode(instr)) {
        __ cmpq(lhs_register, i.MemoryOperand(kValueIndex));
      } else {
        __ cmpq(lhs_register, i.InputRegister(kValueIndex));
      }
      break;
    }
    case kArchStackCheckOffset:
      __ Move(i.OutputRegister(), Smi::FromInt(GetStackCheckOffset()));
      break;
    case kArchTruncateDoubleToI: {
      auto result = i.OutputRegister();
      auto input = i.InputDoubleRegister(0);
      auto ool = zone()->New<OutOfLineTruncateDoubleToI>(
          this, result, input, DetermineStubCallMode(),
          &unwinding_info_writer_);
      // We use Cvttsd2siq instead of Cvttsd2si due to performance reasons. The
      // use of Cvttsd2siq requires the movl below to avoid sign extension.
      __ Cvttsd2siq(result, input);
      __ cmpq(result, Immediate(1));
      __ j(overflow, ool->entry());
      __ bind(ool->exit());
      __ movl(result, result);
      break;
    }
    case kArchStoreWithWriteBarrier: {
      RecordWriteMode mode =
          static_cast<RecordWriteMode>(MiscField::decode(instr->opcode()));
      Register object = i.InputRegister(0);
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      Register value = i.InputRegister(index);
      Register scratch0 = i.TempRegister(0);
      Register scratch1 = i.TempRegister(1);
      auto ool = zone()->New<OutOfLineRecordWrite>(this, object, operand, value,
                                                   scratch0, scratch1, mode,
                                                   DetermineStubCallMode());
      __ StoreTaggedField(operand, value);
      __ CheckPageFlag(object, scratch0,
                       MemoryChunk::kPointersFromHereAreInterestingMask,
                       not_zero, ool->entry());
      __ bind(ool->exit());
      break;
    }
    case kArchWordPoisonOnSpeculation:
      DCHECK_EQ(i.OutputRegister(), i.InputRegister(0));
      __ andq(i.InputRegister(0), kSpeculationPoisonRegister);
      break;
    case kX64MFence:
      __ mfence();
      break;
    case kX64LFence:
      __ lfence();
      break;
    case kArchStackSlot: {
      FrameOffset offset =
          frame_access_state()->GetFrameOffset(i.InputInt32(0));
      Register base = offset.from_stack_pointer() ? rsp : rbp;
      __ leaq(i.OutputRegister(), Operand(base, offset.offset()));
      break;
    }
    case kIeee754Float64Acos:
      ASSEMBLE_IEEE754_UNOP(acos);
      break;
    case kIeee754Float64Acosh:
      ASSEMBLE_IEEE754_UNOP(acosh);
      break;
    case kIeee754Float64Asin:
      ASSEMBLE_IEEE754_UNOP(asin);
      break;
    case kIeee754Float64Asinh:
      ASSEMBLE_IEEE754_UNOP(asinh);
      break;
    case kIeee754Float64Atan:
      ASSEMBLE_IEEE754_UNOP(atan);
      break;
    case kIeee754Float64Atanh:
      ASSEMBLE_IEEE754_UNOP(atanh);
      break;
    case kIeee754Float64Atan2:
      ASSEMBLE_IEEE754_BINOP(atan2);
      break;
    case kIeee754Float64Cbrt:
      ASSEMBLE_IEEE754_UNOP(cbrt);
      break;
    case kIeee754Float64Cos:
      ASSEMBLE_IEEE754_UNOP(cos);
      break;
    case kIeee754Float64Cosh:
      ASSEMBLE_IEEE754_UNOP(cosh);
      break;
    case kIeee754Float64Exp:
      ASSEMBLE_IEEE754_UNOP(exp);
      break;
    case kIeee754Float64Expm1:
      ASSEMBLE_IEEE754_UNOP(expm1);
      break;
    case kIeee754Float64Log:
      ASSEMBLE_IEEE754_UNOP(log);
      break;
    case kIeee754Float64Log1p:
      ASSEMBLE_IEEE754_UNOP(log1p);
      break;
    case kIeee754Float64Log2:
      ASSEMBLE_IEEE754_UNOP(log2);
      break;
    case kIeee754Float64Log10:
      ASSEMBLE_IEEE754_UNOP(log10);
      break;
    case kIeee754Float64Pow:
      ASSEMBLE_IEEE754_BINOP(pow);
      break;
    case kIeee754Float64Sin:
      ASSEMBLE_IEEE754_UNOP(sin);
      break;
    case kIeee754Float64Sinh:
      ASSEMBLE_IEEE754_UNOP(sinh);
      break;
    case kIeee754Float64Tan:
      ASSEMBLE_IEEE754_UNOP(tan);
      break;
    case kIeee754Float64Tanh:
      ASSEMBLE_IEEE754_UNOP(tanh);
      break;
    case kX64Add32:
      ASSEMBLE_BINOP(addl);
      break;
    case kX64Add:
      ASSEMBLE_BINOP(addq);
      break;
    case kX64Sub32:
      ASSEMBLE_BINOP(subl);
      break;
    case kX64Sub:
      ASSEMBLE_BINOP(subq);
      break;
    case kX64And32:
      ASSEMBLE_BINOP(andl);
      break;
    case kX64And:
      ASSEMBLE_BINOP(andq);
      break;
    case kX64Cmp8:
      ASSEMBLE_COMPARE(cmpb);
      break;
    case kX64Cmp16:
      ASSEMBLE_COMPARE(cmpw);
      break;
    case kX64Cmp32:
      ASSEMBLE_COMPARE(cmpl);
      break;
    case kX64Cmp:
      ASSEMBLE_COMPARE(cmpq);
      break;
    case kX64Test8:
      ASSEMBLE_COMPARE(testb);
      break;
    case kX64Test16:
      ASSEMBLE_COMPARE(testw);
      break;
    case kX64Test32:
      ASSEMBLE_COMPARE(testl);
      break;
    case kX64Test:
      ASSEMBLE_COMPARE(testq);
      break;
    case kX64Imul32:
      ASSEMBLE_MULT(imull);
      break;
    case kX64Imul:
      ASSEMBLE_MULT(imulq);
      break;
    case kX64ImulHigh32:
      if (HasRegisterInput(instr, 1)) {
        __ imull(i.InputRegister(1));
      } else {
        __ imull(i.InputOperand(1));
      }
      break;
    case kX64UmulHigh32:
      if (HasRegisterInput(instr, 1)) {
        __ mull(i.InputRegister(1));
      } else {
        __ mull(i.InputOperand(1));
      }
      break;
    case kX64Idiv32:
      __ cdq();
      __ idivl(i.InputRegister(1));
      break;
    case kX64Idiv:
      __ cqo();
      __ idivq(i.InputRegister(1));
      break;
    case kX64Udiv32:
      __ xorl(rdx, rdx);
      __ divl(i.InputRegister(1));
      break;
    case kX64Udiv:
      __ xorq(rdx, rdx);
      __ divq(i.InputRegister(1));
      break;
    case kX64Not:
      ASSEMBLE_UNOP(notq);
      break;
    case kX64Not32:
      ASSEMBLE_UNOP(notl);
      break;
    case kX64Neg:
      ASSEMBLE_UNOP(negq);
      break;
    case kX64Neg32:
      ASSEMBLE_UNOP(negl);
      break;
    case kX64Or32:
      ASSEMBLE_BINOP(orl);
      break;
    case kX64Or:
      ASSEMBLE_BINOP(orq);
      break;
    case kX64Xor32:
      ASSEMBLE_BINOP(xorl);
      break;
    case kX64Xor:
      ASSEMBLE_BINOP(xorq);
      break;
    case kX64Shl32:
      ASSEMBLE_SHIFT(shll, 5);
      break;
    case kX64Shl:
      ASSEMBLE_SHIFT(shlq, 6);
      break;
    case kX64Shr32:
      ASSEMBLE_SHIFT(shrl, 5);
      break;
    case kX64Shr:
      ASSEMBLE_SHIFT(shrq, 6);
      break;
    case kX64Sar32:
      ASSEMBLE_SHIFT(sarl, 5);
      break;
    case kX64Sar:
      ASSEMBLE_SHIFT(sarq, 6);
      break;
    case kX64Rol32:
      ASSEMBLE_SHIFT(roll, 5);
      break;
    case kX64Rol:
      ASSEMBLE_SHIFT(rolq, 6);
      break;
    case kX64Ror32:
      ASSEMBLE_SHIFT(rorl, 5);
      break;
    case kX64Ror:
      ASSEMBLE_SHIFT(rorq, 6);
      break;
    case kX64Lzcnt:
      if (HasRegisterInput(instr, 0)) {
        __ Lzcntq(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Lzcntq(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Lzcnt32:
      if (HasRegisterInput(instr, 0)) {
        __ Lzcntl(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Lzcntl(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Tzcnt:
      if (HasRegisterInput(instr, 0)) {
        __ Tzcntq(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Tzcntq(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Tzcnt32:
      if (HasRegisterInput(instr, 0)) {
        __ Tzcntl(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Tzcntl(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Popcnt:
      if (HasRegisterInput(instr, 0)) {
        __ Popcntq(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Popcntq(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Popcnt32:
      if (HasRegisterInput(instr, 0)) {
        __ Popcntl(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ Popcntl(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kX64Bswap:
      __ bswapq(i.OutputRegister());
      break;
    case kX64Bswap32:
      __ bswapl(i.OutputRegister());
      break;
    case kSSEFloat32Cmp:
      ASSEMBLE_SSE_BINOP(Ucomiss);
      break;
    case kSSEFloat32Add:
      ASSEMBLE_SSE_BINOP(addss);
      break;
    case kSSEFloat32Sub:
      ASSEMBLE_SSE_BINOP(subss);
      break;
    case kSSEFloat32Mul:
      ASSEMBLE_SSE_BINOP(mulss);
      break;
    case kSSEFloat32Div:
      ASSEMBLE_SSE_BINOP(divss);
      // Don't delete this mov. It may improve performance on some CPUs,
      // when there is a (v)mulss depending on the result.
      __ movaps(i.OutputDoubleRegister(), i.OutputDoubleRegister());
      break;
    case kSSEFloat32Abs: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      XMMRegister tmp = i.ToDoubleRegister(instr->TempAt(0));
      __ Pcmpeqd(tmp, tmp);
      __ Psrlq(tmp, 33);
      __ Andps(i.OutputDoubleRegister(), tmp);
      break;
    }
    case kSSEFloat32Neg: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      XMMRegister tmp = i.ToDoubleRegister(instr->TempAt(0));
      __ Pcmpeqd(tmp, tmp);
      __ Psllq(tmp, 31);
      __ Xorps(i.OutputDoubleRegister(), tmp);
      break;
    }
    case kSSEFloat32Sqrt:
      ASSEMBLE_SSE_UNOP(sqrtss);
      break;
    case kSSEFloat32ToFloat64:
      ASSEMBLE_SSE_UNOP(Cvtss2sd);
      break;
    case kSSEFloat32Round: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      RoundingMode const mode =
          static_cast<RoundingMode>(MiscField::decode(instr->opcode()));
      __ Roundss(i.OutputDoubleRegister(), i.InputDoubleRegister(0), mode);
      break;
    }
    case kSSEFloat32ToInt32:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttss2si(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttss2si(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kSSEFloat32ToUint32: {
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttss2siq(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttss2siq(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    }
    case kSSEFloat64Cmp:
      ASSEMBLE_SSE_BINOP(Ucomisd);
      break;
    case kSSEFloat64Add:
      ASSEMBLE_SSE_BINOP(addsd);
      break;
    case kSSEFloat64Sub:
      ASSEMBLE_SSE_BINOP(subsd);
      break;
    case kSSEFloat64Mul:
      ASSEMBLE_SSE_BINOP(mulsd);
      break;
    case kSSEFloat64Div:
      ASSEMBLE_SSE_BINOP(divsd);
      // Don't delete this mov. It may improve performance on some CPUs,
      // when there is a (v)mulsd depending on the result.
      __ Movapd(i.OutputDoubleRegister(), i.OutputDoubleRegister());
      break;
    case kSSEFloat64Mod: {
      __ AllocateStackSpace(kDoubleSize);
      unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                       kDoubleSize);
      // Move values to st(0) and st(1).
      __ Movsd(Operand(rsp, 0), i.InputDoubleRegister(1));
      __ fld_d(Operand(rsp, 0));
      __ Movsd(Operand(rsp, 0), i.InputDoubleRegister(0));
      __ fld_d(Operand(rsp, 0));
      // Loop while fprem isn't done.
      Label mod_loop;
      __ bind(&mod_loop);
      // This instructions traps on all kinds inputs, but we are assuming the
      // floating point control word is set to ignore them all.
      __ fprem();
      // The following 2 instruction implicitly use rax.
      __ fnstsw_ax();
      if (CpuFeatures::IsSupported(SAHF)) {
        CpuFeatureScope sahf_scope(tasm(), SAHF);
        __ sahf();
      } else {
        __ shrl(rax, Immediate(8));
        __ andl(rax, Immediate(0xFF));
        __ pushq(rax);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kSystemPointerSize);
        __ popfq();
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         -kSystemPointerSize);
      }
      __ j(parity_even, &mod_loop);
      // Move output to stack and clean up.
      __ fstp(1);
      __ fstp_d(Operand(rsp, 0));
      __ Movsd(i.OutputDoubleRegister(), Operand(rsp, 0));
      __ addq(rsp, Immediate(kDoubleSize));
      unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                       -kDoubleSize);
      break;
    }
    case kSSEFloat32Max: {
      Label compare_swap, done_compare;
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Ucomiss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Ucomiss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      auto ool =
          zone()->New<OutOfLineLoadFloat32NaN>(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(above, &done_compare, Label::kNear);
      __ j(below, &compare_swap, Label::kNear);
      __ Movmskps(kScratchRegister, i.InputDoubleRegister(0));
      __ testl(kScratchRegister, Immediate(1));
      __ j(zero, &done_compare, Label::kNear);
      __ bind(&compare_swap);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Movss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      __ bind(&done_compare);
      __ bind(ool->exit());
      break;
    }
    case kSSEFloat32Min: {
      Label compare_swap, done_compare;
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Ucomiss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Ucomiss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      auto ool =
          zone()->New<OutOfLineLoadFloat32NaN>(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(below, &done_compare, Label::kNear);
      __ j(above, &compare_swap, Label::kNear);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movmskps(kScratchRegister, i.InputDoubleRegister(1));
      } else {
        __ Movss(kScratchDoubleReg, i.InputOperand(1));
        __ Movmskps(kScratchRegister, kScratchDoubleReg);
      }
      __ testl(kScratchRegister, Immediate(1));
      __ j(zero, &done_compare, Label::kNear);
      __ bind(&compare_swap);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Movss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      __ bind(&done_compare);
      __ bind(ool->exit());
      break;
    }
    case kSSEFloat64Max: {
      Label compare_swap, done_compare;
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Ucomisd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Ucomisd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      auto ool =
          zone()->New<OutOfLineLoadFloat64NaN>(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(above, &done_compare, Label::kNear);
      __ j(below, &compare_swap, Label::kNear);
      __ Movmskpd(kScratchRegister, i.InputDoubleRegister(0));
      __ testl(kScratchRegister, Immediate(1));
      __ j(zero, &done_compare, Label::kNear);
      __ bind(&compare_swap);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movsd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Movsd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      __ bind(&done_compare);
      __ bind(ool->exit());
      break;
    }
    case kSSEFloat64Min: {
      Label compare_swap, done_compare;
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Ucomisd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Ucomisd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      auto ool =
          zone()->New<OutOfLineLoadFloat64NaN>(this, i.OutputDoubleRegister());
      __ j(parity_even, ool->entry());
      __ j(below, &done_compare, Label::kNear);
      __ j(above, &compare_swap, Label::kNear);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movmskpd(kScratchRegister, i.InputDoubleRegister(1));
      } else {
        __ Movsd(kScratchDoubleReg, i.InputOperand(1));
        __ Movmskpd(kScratchRegister, kScratchDoubleReg);
      }
      __ testl(kScratchRegister, Immediate(1));
      __ j(zero, &done_compare, Label::kNear);
      __ bind(&compare_swap);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ Movsd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ Movsd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      __ bind(&done_compare);
      __ bind(ool->exit());
      break;
    }
    case kX64F64x2Abs:
    case kSSEFloat64Abs: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      XMMRegister tmp = i.ToDoubleRegister(instr->TempAt(0));
      __ Pcmpeqd(tmp, tmp);
      __ Psrlq(tmp, 1);
      __ Andpd(i.OutputDoubleRegister(), tmp);
      break;
    }
    case kX64F64x2Neg:
    case kSSEFloat64Neg: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      XMMRegister tmp = i.ToDoubleRegister(instr->TempAt(0));
      __ Pcmpeqd(tmp, tmp);
      __ Psllq(tmp, 63);
      __ Xorpd(i.OutputDoubleRegister(), tmp);
      break;
    }
    case kSSEFloat64Sqrt:
      ASSEMBLE_SSE_UNOP(Sqrtsd);
      break;
    case kSSEFloat64Round: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      RoundingMode const mode =
          static_cast<RoundingMode>(MiscField::decode(instr->opcode()));
      __ Roundsd(i.OutputDoubleRegister(), i.InputDoubleRegister(0), mode);
      break;
    }
    case kSSEFloat64ToFloat32:
      ASSEMBLE_SSE_UNOP(Cvtsd2ss);
      break;
    case kSSEFloat64ToInt32:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttsd2si(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttsd2si(i.OutputRegister(), i.InputOperand(0));
      }
      break;
    case kSSEFloat64ToUint32: {
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttsd2siq(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttsd2siq(i.OutputRegister(), i.InputOperand(0));
      }
      if (MiscField::decode(instr->opcode())) {
        __ AssertZeroExtended(i.OutputRegister());
      }
      break;
    }
    case kSSEFloat32ToInt64:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttss2siq(i.OutputRegister(), i.InputDoubleRegister(0));
      } else {
        __ Cvttss2siq(i.OutputRegister(), i.InputOperand(0));
      }
      if (instr->OutputCount() > 1) {
        __ Set(i.OutputRegister(1), 1);
        Label done;
        Label fail;
        __ Move(kScratchDoubleReg, static_cast<float>(INT64_MIN));
        if (instr->InputAt(0)->IsFPRegister()) {
          __ Ucomiss(kScratchDoubleReg, i.InputDoubleRegister(0));
        } else {
          __ Ucomiss(kScratchDoubleReg, i.InputOperand(0));
        }
        // If the input is NaN, then the conversion fails.
        __ j(parity_even, &fail, Label::kNear);
        // If the input is INT64_MIN, then the conversion succeeds.
        __ j(equal, &done, Label::kNear);
        __ cmpq(i.OutputRegister(0), Immediate(1));
        // If the conversion results in INT64_MIN, but the input was not
        // INT64_MIN, then the conversion fails.
        __ j(no_overflow, &done, Label::kNear);
        __ bind(&fail);
        __ Set(i.OutputRegister(1), 0);
        __ bind(&done);
      }
      break;
    case kSSEFloat64ToInt64:
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttsd2siq(i.OutputRegister(0), i.InputDoubleRegister(0));
      } else {
        __ Cvttsd2siq(i.OutputRegister(0), i.InputOperand(0));
      }
      if (instr->OutputCount() > 1) {
        __ Set(i.OutputRegister(1), 1);
        Label done;
        Label fail;
        __ Move(kScratchDoubleReg, static_cast<double>(INT64_MIN));
        if (instr->InputAt(0)->IsFPRegister()) {
          __ Ucomisd(kScratchDoubleReg, i.InputDoubleRegister(0));
        } else {
          __ Ucomisd(kScratchDoubleReg, i.InputOperand(0));
        }
        // If the input is NaN, then the conversion fails.
        __ j(parity_even, &fail, Label::kNear);
        // If the input is INT64_MIN, then the conversion succeeds.
        __ j(equal, &done, Label::kNear);
        __ cmpq(i.OutputRegister(0), Immediate(1));
        // If the conversion results in INT64_MIN, but the input was not
        // INT64_MIN, then the conversion fails.
        __ j(no_overflow, &done, Label::kNear);
        __ bind(&fail);
        __ Set(i.OutputRegister(1), 0);
        __ bind(&done);
      }
      break;
    case kSSEFloat32ToUint64: {
      Label fail;
      if (instr->OutputCount() > 1) __ Set(i.OutputRegister(1), 0);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttss2uiq(i.OutputRegister(), i.InputDoubleRegister(0), &fail);
      } else {
        __ Cvttss2uiq(i.OutputRegister(), i.InputOperand(0), &fail);
      }
      if (instr->OutputCount() > 1) __ Set(i.OutputRegister(1), 1);
      __ bind(&fail);
      break;
    }
    case kSSEFloat64ToUint64: {
      Label fail;
      if (instr->OutputCount() > 1) __ Set(i.OutputRegister(1), 0);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Cvttsd2uiq(i.OutputRegister(), i.InputDoubleRegister(0), &fail);
      } else {
        __ Cvttsd2uiq(i.OutputRegister(), i.InputOperand(0), &fail);
      }
      if (instr->OutputCount() > 1) __ Set(i.OutputRegister(1), 1);
      __ bind(&fail);
      break;
    }
    case kSSEInt32ToFloat64:
      if (HasRegisterInput(instr, 0)) {
        __ Cvtlsi2sd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtlsi2sd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEInt32ToFloat32:
      if (HasRegisterInput(instr, 0)) {
        __ Cvtlsi2ss(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtlsi2ss(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEInt64ToFloat32:
      if (HasRegisterInput(instr, 0)) {
        __ Cvtqsi2ss(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtqsi2ss(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEInt64ToFloat64:
      if (HasRegisterInput(instr, 0)) {
        __ Cvtqsi2sd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtqsi2sd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEUint64ToFloat32:
      if (HasRegisterInput(instr, 0)) {
        __ Cvtqui2ss(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtqui2ss(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEUint64ToFloat64:
      if (HasRegisterInput(instr, 0)) {
        __ Cvtqui2sd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtqui2sd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEUint32ToFloat64:
      if (HasRegisterInput(instr, 0)) {
        __ Cvtlui2sd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtlui2sd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEUint32ToFloat32:
      if (HasRegisterInput(instr, 0)) {
        __ Cvtlui2ss(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Cvtlui2ss(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kSSEFloat64ExtractLowWord32:
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ movl(i.OutputRegister(), i.InputOperand(0));
      } else {
        __ Movd(i.OutputRegister(), i.InputDoubleRegister(0));
      }
      break;
    case kSSEFloat64ExtractHighWord32:
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ movl(i.OutputRegister(), i.InputOperand(0, kDoubleSize / 2));
      } else {
        __ Pextrd(i.OutputRegister(), i.InputDoubleRegister(0), 1);
      }
      break;
    case kSSEFloat64InsertLowWord32:
      if (HasRegisterInput(instr, 1)) {
        __ Pinsrd(i.OutputDoubleRegister(), i.InputRegister(1), 0);
      } else {
        __ Pinsrd(i.OutputDoubleRegister(), i.InputOperand(1), 0);
      }
      break;
    case kSSEFloat64InsertHighWord32:
      if (HasRegisterInput(instr, 1)) {
        __ Pinsrd(i.OutputDoubleRegister(), i.InputRegister(1), 1);
      } else {
        __ Pinsrd(i.OutputDoubleRegister(), i.InputOperand(1), 1);
      }
      break;
    case kSSEFloat64LoadLowWord32:
      if (HasRegisterInput(instr, 0)) {
        __ Movd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Movd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kAVXFloat32Cmp: {
      CpuFeatureScope avx_scope(tasm(), AVX);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ vucomiss(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ vucomiss(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      break;
    }
    case kAVXFloat32Add:
      ASSEMBLE_AVX_BINOP(vaddss);
      break;
    case kAVXFloat32Sub:
      ASSEMBLE_AVX_BINOP(vsubss);
      break;
    case kAVXFloat32Mul:
      ASSEMBLE_AVX_BINOP(vmulss);
      break;
    case kAVXFloat32Div:
      ASSEMBLE_AVX_BINOP(vdivss);
      // Don't delete this mov. It may improve performance on some CPUs,
      // when there is a (v)mulss depending on the result.
      __ Movaps(i.OutputDoubleRegister(), i.OutputDoubleRegister());
      break;
    case kAVXFloat64Cmp: {
      CpuFeatureScope avx_scope(tasm(), AVX);
      if (instr->InputAt(1)->IsFPRegister()) {
        __ vucomisd(i.InputDoubleRegister(0), i.InputDoubleRegister(1));
      } else {
        __ vucomisd(i.InputDoubleRegister(0), i.InputOperand(1));
      }
      break;
    }
    case kAVXFloat64Add:
      ASSEMBLE_AVX_BINOP(vaddsd);
      break;
    case kAVXFloat64Sub:
      ASSEMBLE_AVX_BINOP(vsubsd);
      break;
    case kAVXFloat64Mul:
      ASSEMBLE_AVX_BINOP(vmulsd);
      break;
    case kAVXFloat64Div:
      ASSEMBLE_AVX_BINOP(vdivsd);
      // Don't delete this mov. It may improve performance on some CPUs,
      // when there is a (v)mulsd depending on the result.
      __ Movapd(i.OutputDoubleRegister(), i.OutputDoubleRegister());
      break;
    case kAVXFloat32Abs: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      CpuFeatureScope avx_scope(tasm(), AVX);
      XMMRegister tmp = i.ToDoubleRegister(instr->TempAt(0));
      __ vpcmpeqd(tmp, tmp, tmp);
      __ vpsrlq(tmp, tmp, 33);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ vandps(i.OutputDoubleRegister(), tmp, i.InputDoubleRegister(0));
      } else {
        __ vandps(i.OutputDoubleRegister(), tmp, i.InputOperand(0));
      }
      break;
    }
    case kAVXFloat32Neg: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      CpuFeatureScope avx_scope(tasm(), AVX);
      XMMRegister tmp = i.ToDoubleRegister(instr->TempAt(0));
      __ vpcmpeqd(tmp, tmp, tmp);
      __ vpsllq(tmp, tmp, 31);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ vxorps(i.OutputDoubleRegister(), tmp, i.InputDoubleRegister(0));
      } else {
        __ vxorps(i.OutputDoubleRegister(), tmp, i.InputOperand(0));
      }
      break;
    }
    case kAVXFloat64Abs: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      CpuFeatureScope avx_scope(tasm(), AVX);
      XMMRegister tmp = i.ToDoubleRegister(instr->TempAt(0));
      __ vpcmpeqd(tmp, tmp, tmp);
      __ vpsrlq(tmp, tmp, 1);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ vandpd(i.OutputDoubleRegister(), tmp, i.InputDoubleRegister(0));
      } else {
        __ vandpd(i.OutputDoubleRegister(), tmp, i.InputOperand(0));
      }
      break;
    }
    case kAVXFloat64Neg: {
      // TODO(bmeurer): Use RIP relative 128-bit constants.
      CpuFeatureScope avx_scope(tasm(), AVX);
      XMMRegister tmp = i.ToDoubleRegister(instr->TempAt(0));
      __ vpcmpeqd(tmp, tmp, tmp);
      __ vpsllq(tmp, tmp, 63);
      if (instr->InputAt(0)->IsFPRegister()) {
        __ vxorpd(i.OutputDoubleRegister(), tmp, i.InputDoubleRegister(0));
      } else {
        __ vxorpd(i.OutputDoubleRegister(), tmp, i.InputOperand(0));
      }
      break;
    }
    case kSSEFloat64SilenceNaN:
      __ Xorpd(kScratchDoubleReg, kScratchDoubleReg);
      __ Subsd(i.InputDoubleRegister(0), kScratchDoubleReg);
      break;
    case kX64Movsxbl:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movsxbl);
      __ AssertZeroExtended(i.OutputRegister());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movzxbl:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movzxbl);
      __ AssertZeroExtended(i.OutputRegister());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movsxbq:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movsxbq);
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movzxbq:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movzxbq);
      __ AssertZeroExtended(i.OutputRegister());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movb: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      if (HasImmediateInput(instr, index)) {
        __ movb(operand, Immediate(i.InputInt8(index)));
      } else {
        __ movb(operand, i.InputRegister(index));
      }
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    }
    case kX64Movsxwl:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movsxwl);
      __ AssertZeroExtended(i.OutputRegister());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movzxwl:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movzxwl);
      __ AssertZeroExtended(i.OutputRegister());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movsxwq:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movsxwq);
      break;
    case kX64Movzxwq:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movzxwq);
      __ AssertZeroExtended(i.OutputRegister());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movw: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      if (HasImmediateInput(instr, index)) {
        __ movw(operand, Immediate(i.InputInt16(index)));
      } else {
        __ movw(operand, i.InputRegister(index));
      }
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    }
    case kX64Movl:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      if (instr->HasOutput()) {
        if (HasAddressingMode(instr)) {
          __ movl(i.OutputRegister(), i.MemoryOperand());
        } else {
          if (HasRegisterInput(instr, 0)) {
            __ movl(i.OutputRegister(), i.InputRegister(0));
          } else {
            __ movl(i.OutputRegister(), i.InputOperand(0));
          }
        }
        __ AssertZeroExtended(i.OutputRegister());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        if (HasImmediateInput(instr, index)) {
          __ movl(operand, i.InputImmediate(index));
        } else {
          __ movl(operand, i.InputRegister(index));
        }
      }
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movsxlq:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      ASSEMBLE_MOVX(movsxlq);
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64MovqDecompressTaggedSigned: {
      CHECK(instr->HasOutput());
      __ DecompressTaggedSigned(i.OutputRegister(), i.MemoryOperand());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    }
    case kX64MovqDecompressTaggedPointer: {
      CHECK(instr->HasOutput());
      __ DecompressTaggedPointer(i.OutputRegister(), i.MemoryOperand());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    }
    case kX64MovqDecompressAnyTagged: {
      CHECK(instr->HasOutput());
      __ DecompressAnyTagged(i.OutputRegister(), i.MemoryOperand());
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    }
    case kX64MovqCompressTagged: {
      CHECK(!instr->HasOutput());
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      if (HasImmediateInput(instr, index)) {
        __ StoreTaggedField(operand, i.InputImmediate(index));
      } else {
        __ StoreTaggedField(operand, i.InputRegister(index));
      }
      break;
    }
    case kX64Movq:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      if (instr->HasOutput()) {
        __ movq(i.OutputRegister(), i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        if (HasImmediateInput(instr, index)) {
          __ movq(operand, i.InputImmediate(index));
        } else {
          __ movq(operand, i.InputRegister(index));
        }
      }
      EmitWordLoadPoisoningIfNeeded(this, opcode, instr, i);
      break;
    case kX64Movss:
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      if (instr->HasOutput()) {
        __ Movss(i.OutputDoubleRegister(), i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        __ Movss(operand, i.InputDoubleRegister(index));
      }
      break;
    case kX64Movsd: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      if (instr->HasOutput()) {
        const MemoryAccessMode access_mode =
            static_cast<MemoryAccessMode>(MiscField::decode(opcode));
        if (access_mode == kMemoryAccessPoisoned) {
          // If we have to poison the loaded value, we load into a general
          // purpose register first, mask it with the poison, and move the
          // value from the general purpose register into the double register.
          __ movq(kScratchRegister, i.MemoryOperand());
          __ andq(kScratchRegister, kSpeculationPoisonRegister);
          __ Movq(i.OutputDoubleRegister(), kScratchRegister);
        } else {
          __ Movsd(i.OutputDoubleRegister(), i.MemoryOperand());
        }
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        __ Movsd(operand, i.InputDoubleRegister(index));
      }
      break;
    }
    case kX64Movdqu: {
      CpuFeatureScope sse_scope(tasm(), SSSE3);
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      if (instr->HasOutput()) {
        __ Movdqu(i.OutputSimd128Register(), i.MemoryOperand());
      } else {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        __ Movdqu(operand, i.InputSimd128Register(index));
      }
      break;
    }
    case kX64BitcastFI:
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ movl(i.OutputRegister(), i.InputOperand(0));
      } else {
        __ Movd(i.OutputRegister(), i.InputDoubleRegister(0));
      }
      break;
    case kX64BitcastDL:
      if (instr->InputAt(0)->IsFPStackSlot()) {
        __ movq(i.OutputRegister(), i.InputOperand(0));
      } else {
        __ Movq(i.OutputRegister(), i.InputDoubleRegister(0));
      }
      break;
    case kX64BitcastIF:
      if (HasRegisterInput(instr, 0)) {
        __ Movd(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Movss(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kX64BitcastLD:
      if (HasRegisterInput(instr, 0)) {
        __ Movq(i.OutputDoubleRegister(), i.InputRegister(0));
      } else {
        __ Movsd(i.OutputDoubleRegister(), i.InputOperand(0));
      }
      break;
    case kX64Lea32: {
      AddressingMode mode = AddressingModeField::decode(instr->opcode());
      // Shorten "leal" to "addl", "subl" or "shll" if the register allocation
      // and addressing mode just happens to work out. The "addl"/"subl" forms
      // in these cases are faster based on measurements.
      if (i.InputRegister(0) == i.OutputRegister()) {
        if (mode == kMode_MRI) {
          int32_t constant_summand = i.InputInt32(1);
          DCHECK_NE(0, constant_summand);
          if (constant_summand > 0) {
            __ addl(i.OutputRegister(), Immediate(constant_summand));
          } else {
            __ subl(i.OutputRegister(),
                    Immediate(base::NegateWithWraparound(constant_summand)));
          }
        } else if (mode == kMode_MR1) {
          if (i.InputRegister(1) == i.OutputRegister()) {
            __ shll(i.OutputRegister(), Immediate(1));
          } else {
            __ addl(i.OutputRegister(), i.InputRegister(1));
          }
        } else if (mode == kMode_M2) {
          __ shll(i.OutputRegister(), Immediate(1));
        } else if (mode == kMode_M4) {
          __ shll(i.OutputRegister(), Immediate(2));
        } else if (mode == kMode_M8) {
          __ shll(i.OutputRegister(), Immediate(3));
        } else {
          __ leal(i.OutputRegister(), i.MemoryOperand());
        }
      } else if (mode == kMode_MR1 &&
                 i.InputRegister(1) == i.OutputRegister()) {
        __ addl(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ leal(i.OutputRegister(), i.MemoryOperand());
      }
      __ AssertZeroExtended(i.OutputRegister());
      break;
    }
    case kX64Lea: {
      AddressingMode mode = AddressingModeField::decode(instr->opcode());
      // Shorten "leaq" to "addq", "subq" or "shlq" if the register allocation
      // and addressing mode just happens to work out. The "addq"/"subq" forms
      // in these cases are faster based on measurements.
      if (i.InputRegister(0) == i.OutputRegister()) {
        if (mode == kMode_MRI) {
          int32_t constant_summand = i.InputInt32(1);
          if (constant_summand > 0) {
            __ addq(i.OutputRegister(), Immediate(constant_summand));
          } else if (constant_summand < 0) {
            __ subq(i.OutputRegister(), Immediate(-constant_summand));
          }
        } else if (mode == kMode_MR1) {
          if (i.InputRegister(1) == i.OutputRegister()) {
            __ shlq(i.OutputRegister(), Immediate(1));
          } else {
            __ addq(i.OutputRegister(), i.InputRegister(1));
          }
        } else if (mode == kMode_M2) {
          __ shlq(i.OutputRegister(), Immediate(1));
        } else if (mode == kMode_M4) {
          __ shlq(i.OutputRegister(), Immediate(2));
        } else if (mode == kMode_M8) {
          __ shlq(i.OutputRegister(), Immediate(3));
        } else {
          __ leaq(i.OutputRegister(), i.MemoryOperand());
        }
      } else if (mode == kMode_MR1 &&
                 i.InputRegister(1) == i.OutputRegister()) {
        __ addq(i.OutputRegister(), i.InputRegister(0));
      } else {
        __ leaq(i.OutputRegister(), i.MemoryOperand());
      }
      break;
    }
    case kX64Dec32:
      __ decl(i.OutputRegister());
      break;
    case kX64Inc32:
      __ incl(i.OutputRegister());
      break;
    case kX64Push:
      if (HasAddressingMode(instr)) {
        size_t index = 0;
        Operand operand = i.MemoryOperand(&index);
        __ pushq(operand);
        frame_access_state()->IncreaseSPDelta(1);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kSystemPointerSize);
      } else if (HasImmediateInput(instr, 0)) {
        __ pushq(i.InputImmediate(0));
        frame_access_state()->IncreaseSPDelta(1);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kSystemPointerSize);
      } else if (HasRegisterInput(instr, 0)) {
        __ pushq(i.InputRegister(0));
        frame_access_state()->IncreaseSPDelta(1);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kSystemPointerSize);
      } else if (instr->InputAt(0)->IsFloatRegister() ||
                 instr->InputAt(0)->IsDoubleRegister()) {
        // TODO(titzer): use another machine instruction?
        __ AllocateStackSpace(kDoubleSize);
        frame_access_state()->IncreaseSPDelta(kDoubleSize / kSystemPointerSize);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kDoubleSize);
        __ Movsd(Operand(rsp, 0), i.InputDoubleRegister(0));
      } else if (instr->InputAt(0)->IsSimd128Register()) {
        // TODO(titzer): use another machine instruction?
        __ AllocateStackSpace(kSimd128Size);
        frame_access_state()->IncreaseSPDelta(kSimd128Size /
                                              kSystemPointerSize);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kSimd128Size);
        __ Movups(Operand(rsp, 0), i.InputSimd128Register(0));
      } else if (instr->InputAt(0)->IsStackSlot() ||
                 instr->InputAt(0)->IsFloatStackSlot() ||
                 instr->InputAt(0)->IsDoubleStackSlot()) {
        __ pushq(i.InputOperand(0));
        frame_access_state()->IncreaseSPDelta(1);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kSystemPointerSize);
      } else {
        DCHECK(instr->InputAt(0)->IsSimd128StackSlot());
        __ Movups(kScratchDoubleReg, i.InputOperand(0));
        // TODO(titzer): use another machine instruction?
        __ AllocateStackSpace(kSimd128Size);
        frame_access_state()->IncreaseSPDelta(kSimd128Size /
                                              kSystemPointerSize);
        unwinding_info_writer_.MaybeIncreaseBaseOffsetAt(__ pc_offset(),
                                                         kSimd128Size);
        __ Movups(Operand(rsp, 0), kScratchDoubleReg);
      }
      break;
    case kX64Poke: {
      int slot = MiscField::decode(instr->opcode());
      if (HasImmediateInput(instr, 0)) {
        __ movq(Operand(rsp, slot * kSystemPointerSize), i.InputImmediate(0));
      } else if (instr->InputAt(0)->IsFPRegister()) {
        LocationOperand* op = LocationOperand::cast(instr->InputAt(0));
        if (op->representation() == MachineRepresentation::kFloat64) {
          __ Movsd(Operand(rsp, slot * kSystemPointerSize),
                   i.InputDoubleRegister(0));
        } else {
          DCHECK_EQ(MachineRepresentation::kFloat32, op->representation());
          __ Movss(Operand(rsp, slot * kSystemPointerSize),
                   i.InputFloatRegister(0));
        }
      } else {
        __ movq(Operand(rsp, slot * kSystemPointerSize), i.InputRegister(0));
      }
      break;
    }
    case kX64Peek: {
      int reverse_slot = i.InputInt32(0);
      int offset =
          FrameSlotToFPOffset(frame()->GetTotalFrameSlotCount() - reverse_slot);
      if (instr->OutputAt(0)->IsFPRegister()) {
        LocationOperand* op = LocationOperand::cast(instr->OutputAt(0));
        if (op->representation() == MachineRepresentation::kFloat64) {
          __ Movsd(i.OutputDoubleRegister(), Operand(rbp, offset));
        } else if (op->representation() == MachineRepresentation::kFloat32) {
          __ Movss(i.OutputFloatRegister(), Operand(rbp, offset));
        } else {
          DCHECK_EQ(MachineRepresentation::kSimd128, op->representation());
          __ Movdqu(i.OutputSimd128Register(), Operand(rbp, offset));
        }
      } else {
        __ movq(i.OutputRegister(), Operand(rbp, offset));
      }
      break;
    }
    case kX64F64x2Splat: {
      XMMRegister dst = i.OutputSimd128Register();
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Movddup(dst, i.InputDoubleRegister(0));
      } else {
        __ Movddup(dst, i.InputOperand(0));
      }
      break;
    }
    case kX64F64x2ExtractLane: {
      __ Pextrq(kScratchRegister, i.InputSimd128Register(0), i.InputInt8(1));
      __ Movq(i.OutputDoubleRegister(), kScratchRegister);
      break;
    }
    case kX64F64x2Sqrt: {
      __ Sqrtpd(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64F64x2Add: {
      ASSEMBLE_SIMD_BINOP(addpd);
      break;
    }
    case kX64F64x2Sub: {
      ASSEMBLE_SIMD_BINOP(subpd);
      break;
    }
    case kX64F64x2Mul: {
      ASSEMBLE_SIMD_BINOP(mulpd);
      break;
    }
    case kX64F64x2Div: {
      ASSEMBLE_SIMD_BINOP(divpd);
      break;
    }
    case kX64F64x2Min: {
      XMMRegister src1 = i.InputSimd128Register(1),
                  dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // The minpd instruction doesn't propagate NaNs and +0's in its first
      // operand. Perform minpd in both orders, merge the resuls, and adjust.
      __ Movapd(kScratchDoubleReg, src1);
      __ Minpd(kScratchDoubleReg, dst);
      __ Minpd(dst, src1);
      // propagate -0's and NaNs, which may be non-canonical.
      __ Orpd(kScratchDoubleReg, dst);
      // Canonicalize NaNs by quieting and clearing the payload.
      __ Cmppd(dst, kScratchDoubleReg, int8_t{3});
      __ Orpd(kScratchDoubleReg, dst);
      __ Psrlq(dst, 13);
      __ Andnpd(dst, kScratchDoubleReg);
      break;
    }
    case kX64F64x2Max: {
      XMMRegister src1 = i.InputSimd128Register(1),
                  dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // The maxpd instruction doesn't propagate NaNs and +0's in its first
      // operand. Perform maxpd in both orders, merge the resuls, and adjust.
      __ Movapd(kScratchDoubleReg, src1);
      __ Maxpd(kScratchDoubleReg, dst);
      __ Maxpd(dst, src1);
      // Find discrepancies.
      __ Xorpd(dst, kScratchDoubleReg);
      // Propagate NaNs, which may be non-canonical.
      __ Orpd(kScratchDoubleReg, dst);
      // Propagate sign discrepancy and (subtle) quiet NaNs.
      __ Subpd(kScratchDoubleReg, dst);
      // Canonicalize NaNs by clearing the payload. Sign is non-deterministic.
      __ Cmppd(dst, kScratchDoubleReg, int8_t{3});
      __ Psrlq(dst, 13);
      __ Andnpd(dst, kScratchDoubleReg);
      break;
    }
    case kX64F64x2Eq: {
      ASSEMBLE_SIMD_BINOP(cmpeqpd);
      break;
    }
    case kX64F64x2Ne: {
      ASSEMBLE_SIMD_BINOP(cmpneqpd);
      break;
    }
    case kX64F64x2Lt: {
      ASSEMBLE_SIMD_BINOP(cmpltpd);
      break;
    }
    case kX64F64x2Le: {
      ASSEMBLE_SIMD_BINOP(cmplepd);
      break;
    }
    case kX64F64x2Qfma: {
      if (CpuFeatures::IsSupported(FMA3)) {
        CpuFeatureScope fma3_scope(tasm(), FMA3);
        __ vfmadd231pd(i.OutputSimd128Register(), i.InputSimd128Register(1),
                       i.InputSimd128Register(2));
      } else {
        XMMRegister tmp = i.TempSimd128Register(0);
        __ Movapd(tmp, i.InputSimd128Register(2));
        __ Mulpd(tmp, i.InputSimd128Register(1));
        __ Addpd(i.OutputSimd128Register(), tmp);
      }
      break;
    }
    case kX64F64x2Qfms: {
      if (CpuFeatures::IsSupported(FMA3)) {
        CpuFeatureScope fma3_scope(tasm(), FMA3);
        __ vfnmadd231pd(i.OutputSimd128Register(), i.InputSimd128Register(1),
                        i.InputSimd128Register(2));
      } else {
        XMMRegister tmp = i.TempSimd128Register(0);
        __ Movapd(tmp, i.InputSimd128Register(2));
        __ Mulpd(tmp, i.InputSimd128Register(1));
        __ Subpd(i.OutputSimd128Register(), tmp);
      }
      break;
    }
    case kX64F32x4Splat: {
      __ Shufps(i.OutputSimd128Register(), i.InputDoubleRegister(0), 0);
      break;
    }
    case kX64F32x4ExtractLane: {
      if (CpuFeatures::IsSupported(AVX)) {
        CpuFeatureScope avx_scope(tasm(), AVX);
        XMMRegister src = i.InputSimd128Register(0);
        // vshufps and leave junk in the 3 high lanes.
        __ vshufps(i.OutputDoubleRegister(), src, src, i.InputInt8(1));
      } else {
        __ extractps(kScratchRegister, i.InputSimd128Register(0),
                     i.InputUint8(1));
        __ movd(i.OutputDoubleRegister(), kScratchRegister);
      }
      break;
    }
    case kX64F32x4ReplaceLane: {
      // The insertps instruction uses imm8[5:4] to indicate the lane
      // that needs to be replaced.
      byte select = i.InputInt8(1) << 4 & 0x30;
      if (instr->InputAt(2)->IsFPRegister()) {
        __ Insertps(i.OutputSimd128Register(), i.InputDoubleRegister(2),
                    select);
      } else {
        __ Insertps(i.OutputSimd128Register(), i.InputOperand(2), select);
      }
      break;
    }
    case kX64F32x4SConvertI32x4: {
      __ Cvtdq2ps(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64F32x4UConvertI32x4: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      DCHECK_NE(i.OutputSimd128Register(), kScratchDoubleReg);
      XMMRegister dst = i.OutputSimd128Register();
      __ Pxor(kScratchDoubleReg, kScratchDoubleReg);  // zeros
      __ Pblendw(kScratchDoubleReg, dst, uint8_t{0x55});  // get lo 16 bits
      __ Psubd(dst, kScratchDoubleReg);                   // get hi 16 bits
      __ Cvtdq2ps(kScratchDoubleReg, kScratchDoubleReg);  // convert lo exactly
      __ Psrld(dst, byte{1});            // divide by 2 to get in unsigned range
      __ Cvtdq2ps(dst, dst);             // convert hi exactly
      __ Addps(dst, dst);                // double hi, exactly
      __ Addps(dst, kScratchDoubleReg);  // add hi and lo, may round.
      break;
    }
    case kX64F32x4Abs: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ Pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
        __ Psrld(kScratchDoubleReg, byte{1});
        __ Andps(i.OutputSimd128Register(), kScratchDoubleReg);
      } else {
        __ Pcmpeqd(dst, dst);
        __ Psrld(dst, byte{1});
        __ Andps(dst, i.InputSimd128Register(0));
      }
      break;
    }
    case kX64F32x4Neg: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ Pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
        __ Pslld(kScratchDoubleReg, byte{31});
        __ Xorps(i.OutputSimd128Register(), kScratchDoubleReg);
      } else {
        __ Pcmpeqd(dst, dst);
        __ Pslld(dst, byte{31});
        __ Xorps(dst, i.InputSimd128Register(0));
      }
      break;
    }
    case kX64F32x4Sqrt: {
      __ Sqrtps(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64F32x4RecipApprox: {
      __ Rcpps(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64F32x4RecipSqrtApprox: {
      __ Rsqrtps(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64F32x4Add: {
      ASSEMBLE_SIMD_BINOP(addps);
      break;
    }
    case kX64F32x4AddHoriz: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Haddps(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F32x4Sub: {
      ASSEMBLE_SIMD_BINOP(subps);
      break;
    }
    case kX64F32x4Mul: {
      ASSEMBLE_SIMD_BINOP(mulps);
      break;
    }
    case kX64F32x4Div: {
      ASSEMBLE_SIMD_BINOP(divps);
      break;
    }
    case kX64F32x4Min: {
      XMMRegister src1 = i.InputSimd128Register(1),
                  dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // The minps instruction doesn't propagate NaNs and +0's in its first
      // operand. Perform minps in both orders, merge the resuls, and adjust.
      __ Movaps(kScratchDoubleReg, src1);
      __ Minps(kScratchDoubleReg, dst);
      __ Minps(dst, src1);
      // propagate -0's and NaNs, which may be non-canonical.
      __ Orps(kScratchDoubleReg, dst);
      // Canonicalize NaNs by quieting and clearing the payload.
      __ Cmpps(dst, kScratchDoubleReg, int8_t{3});
      __ Orps(kScratchDoubleReg, dst);
      __ Psrld(dst, byte{10});
      __ Andnps(dst, kScratchDoubleReg);
      break;
    }
    case kX64F32x4Max: {
      XMMRegister src1 = i.InputSimd128Register(1),
                  dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // The maxps instruction doesn't propagate NaNs and +0's in its first
      // operand. Perform maxps in both orders, merge the resuls, and adjust.
      __ Movaps(kScratchDoubleReg, src1);
      __ Maxps(kScratchDoubleReg, dst);
      __ Maxps(dst, src1);
      // Find discrepancies.
      __ Xorps(dst, kScratchDoubleReg);
      // Propagate NaNs, which may be non-canonical.
      __ Orps(kScratchDoubleReg, dst);
      // Propagate sign discrepancy and (subtle) quiet NaNs.
      __ Subps(kScratchDoubleReg, dst);
      // Canonicalize NaNs by clearing the payload. Sign is non-deterministic.
      __ Cmpps(dst, kScratchDoubleReg, int8_t{3});
      __ Psrld(dst, byte{10});
      __ Andnps(dst, kScratchDoubleReg);
      break;
    }
    case kX64F32x4Eq: {
      ASSEMBLE_SIMD_BINOP(cmpeqps);
      break;
    }
    case kX64F32x4Ne: {
      ASSEMBLE_SIMD_BINOP(cmpneqps);
      break;
    }
    case kX64F32x4Lt: {
      ASSEMBLE_SIMD_BINOP(cmpltps);
      break;
    }
    case kX64F32x4Le: {
      ASSEMBLE_SIMD_BINOP(cmpleps);
      break;
    }
    case kX64F32x4Qfma: {
      if (CpuFeatures::IsSupported(FMA3)) {
        CpuFeatureScope fma3_scope(tasm(), FMA3);
        __ vfmadd231ps(i.OutputSimd128Register(), i.InputSimd128Register(1),
                       i.InputSimd128Register(2));
      } else {
        XMMRegister tmp = i.TempSimd128Register(0);
        __ Movaps(tmp, i.InputSimd128Register(2));
        __ Mulps(tmp, i.InputSimd128Register(1));
        __ Addps(i.OutputSimd128Register(), tmp);
      }
      break;
    }
    case kX64F32x4Qfms: {
      if (CpuFeatures::IsSupported(FMA3)) {
        CpuFeatureScope fma3_scope(tasm(), FMA3);
        __ vfnmadd231ps(i.OutputSimd128Register(), i.InputSimd128Register(1),
                        i.InputSimd128Register(2));
      } else {
        XMMRegister tmp = i.TempSimd128Register(0);
        __ Movaps(tmp, i.InputSimd128Register(2));
        __ Mulps(tmp, i.InputSimd128Register(1));
        __ Subps(i.OutputSimd128Register(), tmp);
      }
      break;
    }
    case kX64F32x4Pmin: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ Minps(dst, i.InputSimd128Register(1));
      break;
    }
    case kX64F32x4Pmax: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ Maxps(dst, i.InputSimd128Register(1));
      break;
    }
    case kX64F32x4Round: {
      RoundingMode const mode =
          static_cast<RoundingMode>(MiscField::decode(instr->opcode()));
      __ Roundps(i.OutputSimd128Register(), i.InputSimd128Register(0), mode);
      break;
    }
    case kX64F64x2Round: {
      RoundingMode const mode =
          static_cast<RoundingMode>(MiscField::decode(instr->opcode()));
      __ Roundpd(i.OutputSimd128Register(), i.InputSimd128Register(0), mode);
      break;
    }
    case kX64F64x2Pmin: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ Minpd(dst, i.InputSimd128Register(1));
      break;
    }
    case kX64F64x2Pmax: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ Maxpd(dst, i.InputSimd128Register(1));
      break;
    }
    case kX64I64x2Splat: {
      XMMRegister dst = i.OutputSimd128Register();
      if (HasRegisterInput(instr, 0)) {
        __ Movq(dst, i.InputRegister(0));
        __ Movddup(dst, dst);
      } else {
        __ Movddup(dst, i.InputOperand(0));
      }
      break;
    }
    case kX64I64x2ExtractLane: {
      __ Pextrq(i.OutputRegister(), i.InputSimd128Register(0), i.InputInt8(1));
      break;
    }
    case kX64I64x2Neg: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ Movapd(kScratchDoubleReg, src);
        src = kScratchDoubleReg;
      }
      __ Pxor(dst, dst);
      __ Psubq(dst, src);
      break;
    }
    case kX64I64x2BitMask: {
      __ Movmskpd(i.OutputRegister(), i.InputSimd128Register(0));
      break;
    }
    case kX64I64x2Shl: {
      // Take shift value modulo 2^6.
      ASSEMBLE_SIMD_SHIFT(psllq, 6);
      break;
    }
    case kX64I64x2ShrS: {
      // TODO(zhin): there is vpsraq but requires AVX512
      // ShrS on each quadword one at a time
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      Register tmp = i.ToRegister(instr->TempAt(0));
      // Modulo 64 not required as sarq_cl will mask cl to 6 bits.

      // lower quadword
      __ Pextrq(tmp, src, int8_t{0x0});
      __ sarq_cl(tmp);
      __ Pinsrq(dst, tmp, uint8_t{0x0});

      // upper quadword
      __ Pextrq(tmp, src, int8_t{0x1});
      __ sarq_cl(tmp);
      __ Pinsrq(dst, tmp, uint8_t{0x1});
      break;
    }
    case kX64I64x2Add: {
      ASSEMBLE_SIMD_BINOP(paddq);
      break;
    }
    case kX64I64x2Sub: {
      ASSEMBLE_SIMD_BINOP(psubq);
      break;
    }
    case kX64I64x2Mul: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister left = i.InputSimd128Register(0);
      XMMRegister right = i.InputSimd128Register(1);
      XMMRegister tmp1 = i.TempSimd128Register(0);
      XMMRegister tmp2 = i.TempSimd128Register(1);

      __ Movaps(tmp1, left);
      __ Movaps(tmp2, right);

      // Multiply high dword of each qword of left with right.
      __ Psrlq(tmp1, 32);
      __ Pmuludq(tmp1, right);

      // Multiply high dword of each qword of right with left.
      __ Psrlq(tmp2, 32);
      __ Pmuludq(tmp2, left);

      __ Paddq(tmp2, tmp1);
      __ Psllq(tmp2, 32);

      __ Pmuludq(left, right);
      __ Paddq(left, tmp2);  // left == dst
      break;
    }
    case kX64I64x2Eq: {
      ASSEMBLE_SIMD_BINOP(pcmpeqq);
      break;
    }
    case kX64I64x2ShrU: {
      // Take shift value modulo 2^6.
      ASSEMBLE_SIMD_SHIFT(psrlq, 6);
      break;
    }
    case kX64I32x4Splat: {
      XMMRegister dst = i.OutputSimd128Register();
      if (HasRegisterInput(instr, 0)) {
        __ Movd(dst, i.InputRegister(0));
      } else {
        // TODO(v8:9198): Pshufd can load from aligned memory once supported.
        __ Movd(dst, i.InputOperand(0));
      }
      __ Pshufd(dst, dst, uint8_t{0x0});
      break;
    }
    case kX64I32x4ExtractLane: {
      __ Pextrd(i.OutputRegister(), i.InputSimd128Register(0), i.InputInt8(1));
      break;
    }
    case kX64I32x4SConvertF32x4: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister tmp = i.TempSimd128Register(0);
      // NAN->0
      __ Movaps(tmp, dst);
      __ Cmpeqps(tmp, tmp);
      __ Pand(dst, tmp);
      // Set top bit if >= 0 (but not -0.0!)
      __ Pxor(tmp, dst);
      // Convert
      __ Cvttps2dq(dst, dst);
      // Set top bit if >=0 is now < 0
      __ Pand(tmp, dst);
      __ Psrad(tmp, byte{31});
      // Set positive overflow lanes to 0x7FFFFFFF
      __ Pxor(dst, tmp);
      break;
    }
    case kX64I32x4SConvertI16x8Low: {
      __ Pmovsxwd(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I32x4SConvertI16x8High: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Palignr(dst, i.InputSimd128Register(0), uint8_t{8});
      __ Pmovsxwd(dst, dst);
      break;
    }
    case kX64I32x4Neg: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ Pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
        __ Psignd(dst, kScratchDoubleReg);
      } else {
        __ Pxor(dst, dst);
        __ Psubd(dst, src);
      }
      break;
    }
    case kX64I32x4Shl: {
      // Take shift value modulo 2^5.
      ASSEMBLE_SIMD_SHIFT(pslld, 5);
      break;
    }
    case kX64I32x4ShrS: {
      // Take shift value modulo 2^5.
      ASSEMBLE_SIMD_SHIFT(psrad, 5);
      break;
    }
    case kX64I32x4Add: {
      ASSEMBLE_SIMD_BINOP(paddd);
      break;
    }
    case kX64I32x4AddHoriz: {
      ASSEMBLE_SIMD_BINOP(phaddd);
      break;
    }
    case kX64I32x4Sub: {
      ASSEMBLE_SIMD_BINOP(psubd);
      break;
    }
    case kX64I32x4Mul: {
      ASSEMBLE_SIMD_BINOP(pmulld);
      break;
    }
    case kX64I32x4MinS: {
      ASSEMBLE_SIMD_BINOP(pminsd);
      break;
    }
    case kX64I32x4MaxS: {
      ASSEMBLE_SIMD_BINOP(pmaxsd);
      break;
    }
    case kX64I32x4Eq: {
      ASSEMBLE_SIMD_BINOP(pcmpeqd);
      break;
    }
    case kX64I32x4Ne: {
      XMMRegister tmp = i.TempSimd128Register(0);
      __ Pcmpeqd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      __ Pcmpeqd(tmp, tmp);
      __ Pxor(i.OutputSimd128Register(), tmp);
      break;
    }
    case kX64I32x4GtS: {
      ASSEMBLE_SIMD_BINOP(pcmpgtd);
      break;
    }
    case kX64I32x4GeS: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ Pminsd(dst, src);
      __ Pcmpeqd(dst, src);
      break;
    }
    case kX64I32x4UConvertF32x4: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister tmp = i.TempSimd128Register(0);
      XMMRegister tmp2 = i.TempSimd128Register(1);
      // NAN->0, negative->0
      __ Pxor(tmp2, tmp2);
      __ Maxps(dst, tmp2);
      // scratch: float representation of max_signed
      __ Pcmpeqd(tmp2, tmp2);
      __ Psrld(tmp2, uint8_t{1});  // 0x7fffffff
      __ Cvtdq2ps(tmp2, tmp2);     // 0x4f000000
      // tmp: convert (src-max_signed).
      // Positive overflow lanes -> 0x7FFFFFFF
      // Negative lanes -> 0
      __ Movaps(tmp, dst);
      __ Subps(tmp, tmp2);
      __ Cmpleps(tmp2, tmp);
      __ Cvttps2dq(tmp, tmp);
      __ Pxor(tmp, tmp2);
      __ Pxor(tmp2, tmp2);
      __ Pmaxsd(tmp, tmp2);
      // convert. Overflow lanes above max_signed will be 0x80000000
      __ Cvttps2dq(dst, dst);
      // Add (src-max_signed) for overflow lanes.
      __ Paddd(dst, tmp);
      break;
    }
    case kX64I32x4UConvertI16x8Low: {
      __ Pmovzxwd(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I32x4UConvertI16x8High: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Palignr(dst, i.InputSimd128Register(0), uint8_t{8});
      __ Pmovzxwd(dst, dst);
      break;
    }
    case kX64I32x4ShrU: {
      // Take shift value modulo 2^5.
      ASSEMBLE_SIMD_SHIFT(psrld, 5);
      break;
    }
    case kX64I32x4MinU: {
      ASSEMBLE_SIMD_BINOP(pminud);
      break;
    }
    case kX64I32x4MaxU: {
      ASSEMBLE_SIMD_BINOP(pmaxud);
      break;
    }
    case kX64I32x4GtU: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);
      __ Pmaxud(dst, src);
      __ Pcmpeqd(dst, src);
      __ Pcmpeqd(tmp, tmp);
      __ Pxor(dst, tmp);
      break;
    }
    case kX64I32x4GeU: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ Pminud(dst, src);
      __ Pcmpeqd(dst, src);
      break;
    }
    case kX64I32x4Abs: {
      __ Pabsd(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I32x4BitMask: {
      __ Movmskps(i.OutputRegister(), i.InputSimd128Register(0));
      break;
    }
    case kX64I32x4DotI16x8S: {
      ASSEMBLE_SIMD_BINOP(pmaddwd);
      break;
    }
    case kX64S128Const: {
      // Emit code for generic constants as all zeros, or ones cases will be
      // handled separately by the selector.
      XMMRegister dst = i.OutputSimd128Register();
      uint32_t imm[4] = {};
      for (int j = 0; j < 4; j++) {
        imm[j] = i.InputUint32(j);
      }
      SetupSimdImmediateInRegister(tasm(), imm, dst);
      break;
    }
    case kX64S128Zero: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Pxor(dst, dst);
      break;
    }
    case kX64S128AllOnes: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Pcmpeqd(dst, dst);
      break;
    }
    case kX64I16x8Splat: {
      XMMRegister dst = i.OutputSimd128Register();
      if (HasRegisterInput(instr, 0)) {
        __ Movd(dst, i.InputRegister(0));
      } else {
        __ Movd(dst, i.InputOperand(0));
      }
      __ Pshuflw(dst, dst, uint8_t{0x0});
      __ Pshufd(dst, dst, uint8_t{0x0});
      break;
    }
    case kX64I16x8ExtractLaneS: {
      Register dst = i.OutputRegister();
      __ Pextrw(dst, i.InputSimd128Register(0), i.InputUint8(1));
      __ movsxwl(dst, dst);
      break;
    }
    case kX64I16x8SConvertI8x16Low: {
      __ Pmovsxbw(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I16x8SConvertI8x16High: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Palignr(dst, i.InputSimd128Register(0), uint8_t{8});
      __ Pmovsxbw(dst, dst);
      break;
    }
    case kX64I16x8Neg: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ Pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
        __ Psignw(dst, kScratchDoubleReg);
      } else {
        __ Pxor(dst, dst);
        __ Psubw(dst, src);
      }
      break;
    }
    case kX64I16x8Shl: {
      // Take shift value modulo 2^4.
      ASSEMBLE_SIMD_SHIFT(psllw, 4);
      break;
    }
    case kX64I16x8ShrS: {
      // Take shift value modulo 2^4.
      ASSEMBLE_SIMD_SHIFT(psraw, 4);
      break;
    }
    case kX64I16x8SConvertI32x4: {
      ASSEMBLE_SIMD_BINOP(packssdw);
      break;
    }
    case kX64I16x8Add: {
      ASSEMBLE_SIMD_BINOP(paddw);
      break;
    }
    case kX64I16x8AddSatS: {
      ASSEMBLE_SIMD_BINOP(paddsw);
      break;
    }
    case kX64I16x8AddHoriz: {
      ASSEMBLE_SIMD_BINOP(phaddw);
      break;
    }
    case kX64I16x8Sub: {
      ASSEMBLE_SIMD_BINOP(psubw);
      break;
    }
    case kX64I16x8SubSatS: {
      ASSEMBLE_SIMD_BINOP(psubsw);
      break;
    }
    case kX64I16x8Mul: {
      ASSEMBLE_SIMD_BINOP(pmullw);
      break;
    }
    case kX64I16x8MinS: {
      ASSEMBLE_SIMD_BINOP(pminsw);
      break;
    }
    case kX64I16x8MaxS: {
      ASSEMBLE_SIMD_BINOP(pmaxsw);
      break;
    }
    case kX64I16x8Eq: {
      ASSEMBLE_SIMD_BINOP(pcmpeqw);
      break;
    }
    case kX64I16x8Ne: {
      XMMRegister tmp = i.TempSimd128Register(0);
      __ Pcmpeqw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      __ Pcmpeqw(tmp, tmp);
      __ Pxor(i.OutputSimd128Register(), tmp);
      break;
    }
    case kX64I16x8GtS: {
      ASSEMBLE_SIMD_BINOP(pcmpgtw);
      break;
    }
    case kX64I16x8GeS: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ Pminsw(dst, src);
      __ Pcmpeqw(dst, src);
      break;
    }
    case kX64I16x8UConvertI8x16Low: {
      __ Pmovzxbw(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I16x8UConvertI8x16High: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Palignr(dst, i.InputSimd128Register(0), uint8_t{8});
      __ Pmovzxbw(dst, dst);
      break;
    }
    case kX64I16x8ShrU: {
      // Take shift value modulo 2^4.
      ASSEMBLE_SIMD_SHIFT(psrlw, 4);
      break;
    }
    case kX64I16x8UConvertI32x4: {
      ASSEMBLE_SIMD_BINOP(packusdw);
      break;
    }
    case kX64I16x8AddSatU: {
      ASSEMBLE_SIMD_BINOP(paddusw);
      break;
    }
    case kX64I16x8SubSatU: {
      ASSEMBLE_SIMD_BINOP(psubusw);
      break;
    }
    case kX64I16x8MinU: {
      ASSEMBLE_SIMD_BINOP(pminuw);
      break;
    }
    case kX64I16x8MaxU: {
      ASSEMBLE_SIMD_BINOP(pmaxuw);
      break;
    }
    case kX64I16x8GtU: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);
      __ Pmaxuw(dst, src);
      __ Pcmpeqw(dst, src);
      __ Pcmpeqw(tmp, tmp);
      __ Pxor(dst, tmp);
      break;
    }
    case kX64I16x8GeU: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ Pminuw(dst, src);
      __ Pcmpeqw(dst, src);
      break;
    }
    case kX64I16x8RoundingAverageU: {
      ASSEMBLE_SIMD_BINOP(pavgw);
      break;
    }
    case kX64I16x8Abs: {
      __ Pabsw(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I16x8BitMask: {
      Register dst = i.OutputRegister();
      XMMRegister tmp = i.TempSimd128Register(0);
      __ Packsswb(tmp, i.InputSimd128Register(0));
      __ Pmovmskb(dst, tmp);
      __ shrq(dst, Immediate(8));
      break;
    }
    case kX64I8x16Splat: {
      XMMRegister dst = i.OutputSimd128Register();
      if (HasRegisterInput(instr, 0)) {
        __ Movd(dst, i.InputRegister(0));
      } else {
        __ Movd(dst, i.InputOperand(0));
      }
      __ Xorps(kScratchDoubleReg, kScratchDoubleReg);
      __ Pshufb(dst, kScratchDoubleReg);
      break;
    }
    case kX64Pextrb: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      size_t index = 0;
      if (HasAddressingMode(instr)) {
        Operand operand = i.MemoryOperand(&index);
        __ Pextrb(operand, i.InputSimd128Register(index),
                  i.InputUint8(index + 1));
      } else {
        __ Pextrb(i.OutputRegister(), i.InputSimd128Register(0),
                  i.InputUint8(1));
      }
      break;
    }
    case kX64Pextrw: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      size_t index = 0;
      if (HasAddressingMode(instr)) {
        Operand operand = i.MemoryOperand(&index);
        __ Pextrw(operand, i.InputSimd128Register(index),
                  i.InputUint8(index + 1));
      } else {
        __ Pextrw(i.OutputRegister(), i.InputSimd128Register(0),
                  i.InputUint8(1));
      }
      break;
    }
    case kX64I8x16ExtractLaneS: {
      Register dst = i.OutputRegister();
      __ Pextrb(dst, i.InputSimd128Register(0), i.InputUint8(1));
      __ movsxbl(dst, dst);
      break;
    }
    case kX64Pinsrb: {
      ASSEMBLE_PINSR(Pinsrb);
      break;
    }
    case kX64Pinsrw: {
      ASSEMBLE_PINSR(Pinsrw);
      break;
    }
    case kX64Pinsrd: {
      ASSEMBLE_PINSR(Pinsrd);
      break;
    }
    case kX64Pinsrq: {
      ASSEMBLE_PINSR(Pinsrq);
      break;
    }
    case kX64I8x16SConvertI16x8: {
      ASSEMBLE_SIMD_BINOP(packsswb);
      break;
    }
    case kX64I8x16Neg: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ Pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
        __ Psignb(dst, kScratchDoubleReg);
      } else {
        __ Pxor(dst, dst);
        __ Psubb(dst, src);
      }
      break;
    }
    case kX64I8x16Shl: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // Temp registers for shift mask and additional moves to XMM registers.
      Register tmp = i.ToRegister(instr->TempAt(0));
      XMMRegister tmp_simd = i.TempSimd128Register(1);
      if (HasImmediateInput(instr, 1)) {
        // Perform 16-bit shift, then mask away low bits.
        uint8_t shift = i.InputInt3(1);
        __ Psllw(dst, byte{shift});

        uint8_t bmask = static_cast<uint8_t>(0xff << shift);
        uint32_t mask = bmask << 24 | bmask << 16 | bmask << 8 | bmask;
        __ movl(tmp, Immediate(mask));
        __ Movd(tmp_simd, tmp);
        __ Pshufd(tmp_simd, tmp_simd, uint8_t{0});
        __ Pand(dst, tmp_simd);
      } else {
        // Mask off the unwanted bits before word-shifting.
        __ Pcmpeqw(kScratchDoubleReg, kScratchDoubleReg);
        // Take shift value modulo 8.
        __ movq(tmp, i.InputRegister(1));
        __ andq(tmp, Immediate(7));
        __ addq(tmp, Immediate(8));
        __ Movq(tmp_simd, tmp);
        __ Psrlw(kScratchDoubleReg, tmp_simd);
        __ Packuswb(kScratchDoubleReg, kScratchDoubleReg);
        __ Pand(dst, kScratchDoubleReg);
        // TODO(zhin): subq here to avoid asking for another temporary register,
        // examine codegen for other i8x16 shifts, they use less instructions.
        __ subq(tmp, Immediate(8));
        __ Movq(tmp_simd, tmp);
        __ Psllw(dst, tmp_simd);
      }
      break;
    }
    case kX64I8x16ShrS: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (HasImmediateInput(instr, 1)) {
        __ Punpckhbw(kScratchDoubleReg, dst);
        __ Punpcklbw(dst, dst);
        uint8_t shift = i.InputInt3(1) + 8;
        __ Psraw(kScratchDoubleReg, shift);
        __ Psraw(dst, shift);
        __ Packsswb(dst, kScratchDoubleReg);
      } else {
        // Temp registers for shift mask andadditional moves to XMM registers.
        Register tmp = i.ToRegister(instr->TempAt(0));
        XMMRegister tmp_simd = i.TempSimd128Register(1);
        // Unpack the bytes into words, do arithmetic shifts, and repack.
        __ Punpckhbw(kScratchDoubleReg, dst);
        __ Punpcklbw(dst, dst);
        // Prepare shift value
        __ movq(tmp, i.InputRegister(1));
        // Take shift value modulo 8.
        __ andq(tmp, Immediate(7));
        __ addq(tmp, Immediate(8));
        __ Movq(tmp_simd, tmp);
        __ Psraw(kScratchDoubleReg, tmp_simd);
        __ Psraw(dst, tmp_simd);
        __ Packsswb(dst, kScratchDoubleReg);
      }
      break;
    }
    case kX64I8x16Add: {
      ASSEMBLE_SIMD_BINOP(paddb);
      break;
    }
    case kX64I8x16AddSatS: {
      ASSEMBLE_SIMD_BINOP(paddsb);
      break;
    }
    case kX64I8x16Sub: {
      ASSEMBLE_SIMD_BINOP(psubb);
      break;
    }
    case kX64I8x16SubSatS: {
      ASSEMBLE_SIMD_BINOP(psubsb);
      break;
    }
    case kX64I8x16Mul: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      XMMRegister right = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);
      // I16x8 view of I8x16
      // left = AAaa AAaa ... AAaa AAaa
      // right= BBbb BBbb ... BBbb BBbb
      // t = 00AA 00AA ... 00AA 00AA
      // s = 00BB 00BB ... 00BB 00BB
      __ Movaps(tmp, dst);
      __ Movaps(kScratchDoubleReg, right);
      __ Psrlw(tmp, byte{8});
      __ Psrlw(kScratchDoubleReg, byte{8});
      // dst = left * 256
      __ Psllw(dst, byte{8});
      // t = I16x8Mul(t, s)
      //    => __PP __PP ...  __PP  __PP
      __ Pmullw(tmp, kScratchDoubleReg);
      // dst = I16x8Mul(left * 256, right)
      //    => pp__ pp__ ...  pp__  pp__
      __ Pmullw(dst, right);
      // t = I16x8Shl(t, 8)
      //    => PP00 PP00 ...  PP00  PP00
      __ Psllw(tmp, byte{8});
      // dst = I16x8Shr(dst, 8)
      //    => 00pp 00pp ...  00pp  00pp
      __ Psrlw(dst, byte{8});
      // dst = I16x8Or(dst, t)
      //    => PPpp PPpp ...  PPpp  PPpp
      __ Por(dst, tmp);
      break;
    }
    case kX64I8x16MinS: {
      ASSEMBLE_SIMD_BINOP(pminsb);
      break;
    }
    case kX64I8x16MaxS: {
      ASSEMBLE_SIMD_BINOP(pmaxsb);
      break;
    }
    case kX64I8x16Eq: {
      ASSEMBLE_SIMD_BINOP(pcmpeqb);
      break;
    }
    case kX64I8x16Ne: {
      XMMRegister tmp = i.TempSimd128Register(0);
      __ Pcmpeqb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      __ Pcmpeqb(tmp, tmp);
      __ Pxor(i.OutputSimd128Register(), tmp);
      break;
    }
    case kX64I8x16GtS: {
      ASSEMBLE_SIMD_BINOP(pcmpgtb);
      break;
    }
    case kX64I8x16GeS: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ Pminsb(dst, src);
      __ Pcmpeqb(dst, src);
      break;
    }
    case kX64I8x16UConvertI16x8: {
      ASSEMBLE_SIMD_BINOP(packuswb);
      break;
    }
    case kX64I8x16ShrU: {
      XMMRegister dst = i.OutputSimd128Register();
      // Unpack the bytes into words, do logical shifts, and repack.
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // Temp registers for shift mask andadditional moves to XMM registers.
      Register tmp = i.ToRegister(instr->TempAt(0));
      XMMRegister tmp_simd = i.TempSimd128Register(1);
      if (HasImmediateInput(instr, 1)) {
        // Perform 16-bit shift, then mask away high bits.
        uint8_t shift = i.InputInt3(1);
        __ Psrlw(dst, byte{shift});

        uint8_t bmask = 0xff >> shift;
        uint32_t mask = bmask << 24 | bmask << 16 | bmask << 8 | bmask;
        __ movl(tmp, Immediate(mask));
        __ Movd(tmp_simd, tmp);
        __ Pshufd(tmp_simd, tmp_simd, byte{0});
        __ Pand(dst, tmp_simd);
      } else {
        __ Punpckhbw(kScratchDoubleReg, dst);
        __ Punpcklbw(dst, dst);
        // Prepare shift value
        __ movq(tmp, i.InputRegister(1));
        // Take shift value modulo 8.
        __ andq(tmp, Immediate(7));
        __ addq(tmp, Immediate(8));
        __ Movq(tmp_simd, tmp);
        __ Psrlw(kScratchDoubleReg, tmp_simd);
        __ Psrlw(dst, tmp_simd);
        __ Packuswb(dst, kScratchDoubleReg);
      }
      break;
    }
    case kX64I8x16AddSatU: {
      ASSEMBLE_SIMD_BINOP(paddusb);
      break;
    }
    case kX64I8x16SubSatU: {
      ASSEMBLE_SIMD_BINOP(psubusb);
      break;
    }
    case kX64I8x16MinU: {
      ASSEMBLE_SIMD_BINOP(pminub);
      break;
    }
    case kX64I8x16MaxU: {
      ASSEMBLE_SIMD_BINOP(pmaxub);
      break;
    }
    case kX64I8x16GtU: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);
      __ Pmaxub(dst, src);
      __ Pcmpeqb(dst, src);
      __ Pcmpeqb(tmp, tmp);
      __ Pxor(dst, tmp);
      break;
    }
    case kX64I8x16GeU: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ Pminub(dst, src);
      __ Pcmpeqb(dst, src);
      break;
    }
    case kX64I8x16RoundingAverageU: {
      ASSEMBLE_SIMD_BINOP(pavgb);
      break;
    }
    case kX64I8x16Abs: {
      __ Pabsb(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I8x16BitMask: {
      __ Pmovmskb(i.OutputRegister(), i.InputSimd128Register(0));
      break;
    }
    case kX64I8x16SignSelect: {
      __ Pblendvb(i.OutputSimd128Register(), i.InputSimd128Register(0),
                  i.InputSimd128Register(1), i.InputSimd128Register(2));
      break;
    }
    case kX64I16x8SignSelect: {
      if (CpuFeatures::IsSupported(AVX)) {
        CpuFeatureScope avx_scope(tasm(), AVX);
        __ vpsraw(kScratchDoubleReg, i.InputSimd128Register(2), 15);
        __ vpblendvb(i.OutputSimd128Register(), i.InputSimd128Register(0),
                     i.InputSimd128Register(1), kScratchDoubleReg);
      } else {
        DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
        XMMRegister mask = i.InputSimd128Register(2);
        DCHECK_EQ(xmm0, mask);
        __ movapd(kScratchDoubleReg, mask);
        __ pxor(mask, mask);
        __ pcmpgtw(mask, kScratchDoubleReg);
        __ pblendvb(i.OutputSimd128Register(), i.InputSimd128Register(1));
        // Restore mask.
        __ movapd(mask, kScratchDoubleReg);
      }
      break;
    }
    case kX64I32x4SignSelect: {
      __ Blendvps(i.OutputSimd128Register(), i.InputSimd128Register(0),
                  i.InputSimd128Register(1), i.InputSimd128Register(2));
      break;
    }
    case kX64I64x2SignSelect: {
      __ Blendvpd(i.OutputSimd128Register(), i.InputSimd128Register(0),
                  i.InputSimd128Register(1), i.InputSimd128Register(2));
      break;
    }
    case kX64S128And: {
      ASSEMBLE_SIMD_BINOP(pand);
      break;
    }
    case kX64S128Or: {
      ASSEMBLE_SIMD_BINOP(por);
      break;
    }
    case kX64S128Xor: {
      ASSEMBLE_SIMD_BINOP(pxor);
      break;
    }
    case kX64S128Not: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ Movaps(kScratchDoubleReg, dst);
        __ Pcmpeqd(dst, dst);
        __ Pxor(dst, kScratchDoubleReg);
      } else {
        __ Pcmpeqd(dst, dst);
        __ Pxor(dst, src);
      }

      break;
    }
    case kX64S128Select: {
      // Mask used here is stored in dst.
      XMMRegister dst = i.OutputSimd128Register();
      __ Movaps(kScratchDoubleReg, i.InputSimd128Register(1));
      __ Xorps(kScratchDoubleReg, i.InputSimd128Register(2));
      __ Andps(dst, kScratchDoubleReg);
      __ Xorps(dst, i.InputSimd128Register(2));
      break;
    }
    case kX64S128AndNot: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // The inputs have been inverted by instruction selector, so we can call
      // andnps here without any modifications.
      __ Andnps(dst, i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16Swizzle: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister mask = i.TempSimd128Register(0);

      // Out-of-range indices should return 0, add 112 so that any value > 15
      // saturates to 128 (top bit set), so pshufb will zero that lane.
      __ Move(mask, uint32_t{0x70707070});
      __ Pshufd(mask, mask, uint8_t{0x0});
      __ Paddusb(mask, i.InputSimd128Register(1));
      __ Pshufb(dst, mask);
      break;
    }
    case kX64I8x16Shuffle: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister tmp_simd = i.TempSimd128Register(0);
      if (instr->InputCount() == 5) {  // only one input operand
        uint32_t mask[4] = {};
        DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
        for (int j = 4; j > 0; j--) {
          mask[j - 1] = i.InputUint32(j);
        }

        SetupSimdImmediateInRegister(tasm(), mask, tmp_simd);
        __ Pshufb(dst, tmp_simd);
      } else {  // two input operands
        DCHECK_EQ(6, instr->InputCount());
        ASSEMBLE_SIMD_INSTR(Movups, kScratchDoubleReg, 0);
        uint32_t mask1[4] = {};
        for (int j = 5; j > 1; j--) {
          uint32_t lanes = i.InputUint32(j);
          for (int k = 0; k < 32; k += 8) {
            uint8_t lane = lanes >> k;
            mask1[j - 2] |= (lane < kSimd128Size ? lane : 0x80) << k;
          }
        }
        SetupSimdImmediateInRegister(tasm(), mask1, tmp_simd);
        __ Pshufb(kScratchDoubleReg, tmp_simd);
        uint32_t mask2[4] = {};
        if (instr->InputAt(1)->IsSimd128Register()) {
          XMMRegister src1 = i.InputSimd128Register(1);
          if (src1 != dst) __ movups(dst, src1);
        } else {
          __ Movups(dst, i.InputOperand(1));
        }
        for (int j = 5; j > 1; j--) {
          uint32_t lanes = i.InputUint32(j);
          for (int k = 0; k < 32; k += 8) {
            uint8_t lane = lanes >> k;
            mask2[j - 2] |= (lane >= kSimd128Size ? (lane & 0x0F) : 0x80) << k;
          }
        }
        SetupSimdImmediateInRegister(tasm(), mask2, tmp_simd);
        __ Pshufb(dst, tmp_simd);
        __ Por(dst, kScratchDoubleReg);
      }
      break;
    }
    case kX64S128Load8Splat: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      XMMRegister dst = i.OutputSimd128Register();
      __ Pinsrb(dst, dst, i.MemoryOperand(), 0);
      __ Pxor(kScratchDoubleReg, kScratchDoubleReg);
      __ Pshufb(dst, kScratchDoubleReg);
      break;
    }
    case kX64S128Load16Splat: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      XMMRegister dst = i.OutputSimd128Register();
      __ Pinsrw(dst, dst, i.MemoryOperand(), 0);
      __ Pshuflw(dst, dst, uint8_t{0});
      __ Punpcklqdq(dst, dst);
      break;
    }
    case kX64S128Load32Splat: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      if (CpuFeatures::IsSupported(AVX)) {
        CpuFeatureScope avx_scope(tasm(), AVX);
        __ vbroadcastss(i.OutputSimd128Register(), i.MemoryOperand());
      } else {
        __ movss(i.OutputSimd128Register(), i.MemoryOperand());
        __ shufps(i.OutputSimd128Register(), i.OutputSimd128Register(),
                  byte{0});
      }
      break;
    }
    case kX64S128Load64Splat: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Movddup(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64S128Load8x8S: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovsxbw(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64S128Load8x8U: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovzxbw(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64S128Load16x4S: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovsxwd(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64S128Load16x4U: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovzxwd(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64S128Load32x2S: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovsxdq(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64S128Load32x2U: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovzxdq(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64S128Store32Lane: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      uint8_t lane = i.InputUint8(index + 1);
      if (lane == 0) {
        __ Movss(operand, i.InputSimd128Register(index));
      } else {
        DCHECK_GE(3, lane);
        __ Extractps(operand, i.InputSimd128Register(index), lane);
      }
      break;
    }
    case kX64S128Store64Lane: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      size_t index = 0;
      Operand operand = i.MemoryOperand(&index);
      uint8_t lane = i.InputUint8(index + 1);
      if (lane == 0) {
        __ Movlps(operand, i.InputSimd128Register(index));
      } else {
        DCHECK_EQ(1, lane);
        __ Movhps(operand, i.InputSimd128Register(index));
      }
      break;
    }
    case kX64S32x4Swizzle: {
      DCHECK_EQ(2, instr->InputCount());
      ASSEMBLE_SIMD_IMM_INSTR(Pshufd, i.OutputSimd128Register(), 0,
                              i.InputUint8(1));
      break;
    }
    case kX64S32x4Shuffle: {
      DCHECK_EQ(4, instr->InputCount());  // Swizzles should be handled above.
      uint8_t shuffle = i.InputUint8(2);
      DCHECK_NE(0xe4, shuffle);  // A simple blend should be handled below.
      ASSEMBLE_SIMD_IMM_INSTR(Pshufd, kScratchDoubleReg, 1, shuffle);
      ASSEMBLE_SIMD_IMM_INSTR(Pshufd, i.OutputSimd128Register(), 0, shuffle);
      __ Pblendw(i.OutputSimd128Register(), kScratchDoubleReg, i.InputUint8(3));
      break;
    }
    case kX64S16x8Blend: {
      ASSEMBLE_SIMD_IMM_SHUFFLE(Pblendw, i.InputUint8(2));
      break;
    }
    case kX64S16x8HalfShuffle1: {
      XMMRegister dst = i.OutputSimd128Register();
      ASSEMBLE_SIMD_IMM_INSTR(Pshuflw, dst, 0, i.InputUint8(1));
      __ Pshufhw(dst, dst, i.InputUint8(2));
      break;
    }
    case kX64S16x8HalfShuffle2: {
      XMMRegister dst = i.OutputSimd128Register();
      ASSEMBLE_SIMD_IMM_INSTR(Pshuflw, kScratchDoubleReg, 1, i.InputUint8(2));
      __ Pshufhw(kScratchDoubleReg, kScratchDoubleReg, i.InputUint8(3));
      ASSEMBLE_SIMD_IMM_INSTR(Pshuflw, dst, 0, i.InputUint8(2));
      __ Pshufhw(dst, dst, i.InputUint8(3));
      __ Pblendw(dst, kScratchDoubleReg, i.InputUint8(4));
      break;
    }
    case kX64S8x16Alignr: {
      ASSEMBLE_SIMD_IMM_SHUFFLE(Palignr, i.InputUint8(2));
      break;
    }
    case kX64S16x8Dup: {
      XMMRegister dst = i.OutputSimd128Register();
      uint8_t lane = i.InputInt8(1) & 0x7;
      uint8_t lane4 = lane & 0x3;
      uint8_t half_dup = lane4 | (lane4 << 2) | (lane4 << 4) | (lane4 << 6);
      if (lane < 4) {
        ASSEMBLE_SIMD_IMM_INSTR(Pshuflw, dst, 0, half_dup);
        __ Pshufd(dst, dst, uint8_t{0});
      } else {
        ASSEMBLE_SIMD_IMM_INSTR(Pshufhw, dst, 0, half_dup);
        __ Pshufd(dst, dst, uint8_t{0xaa});
      }
      break;
    }
    case kX64S8x16Dup: {
      XMMRegister dst = i.OutputSimd128Register();
      uint8_t lane = i.InputInt8(1) & 0xf;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (lane < 8) {
        __ Punpcklbw(dst, dst);
      } else {
        __ Punpckhbw(dst, dst);
      }
      lane &= 0x7;
      uint8_t lane4 = lane & 0x3;
      uint8_t half_dup = lane4 | (lane4 << 2) | (lane4 << 4) | (lane4 << 6);
      if (lane < 4) {
        __ Pshuflw(dst, dst, half_dup);
        __ Pshufd(dst, dst, uint8_t{0});
      } else {
        __ Pshufhw(dst, dst, half_dup);
        __ Pshufd(dst, dst, uint8_t{0xaa});
      }
      break;
    }
    case kX64S64x2UnpackHigh:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(Punpckhqdq);
      break;
    case kX64S32x4UnpackHigh:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(Punpckhdq);
      break;
    case kX64S16x8UnpackHigh:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(Punpckhwd);
      break;
    case kX64S8x16UnpackHigh:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(Punpckhbw);
      break;
    case kX64S64x2UnpackLow:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(Punpcklqdq);
      break;
    case kX64S32x4UnpackLow:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(Punpckldq);
      break;
    case kX64S16x8UnpackLow:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(Punpcklwd);
      break;
    case kX64S8x16UnpackLow:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(Punpcklbw);
      break;
    case kX64S16x8UnzipHigh: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src2 = dst;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (instr->InputCount() == 2) {
        ASSEMBLE_SIMD_INSTR(Movups, kScratchDoubleReg, 1);
        __ Psrld(kScratchDoubleReg, byte{16});
        src2 = kScratchDoubleReg;
      }
      __ Psrld(dst, byte{16});
      __ Packusdw(dst, src2);
      break;
    }
    case kX64S16x8UnzipLow: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src2 = dst;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ Pxor(kScratchDoubleReg, kScratchDoubleReg);
      if (instr->InputCount() == 2) {
        ASSEMBLE_SIMD_IMM_INSTR(Pblendw, kScratchDoubleReg, 1, uint8_t{0x55});
        src2 = kScratchDoubleReg;
      }
      __ Pblendw(dst, kScratchDoubleReg, uint8_t{0xaa});
      __ Packusdw(dst, src2);
      break;
    }
    case kX64S8x16UnzipHigh: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src2 = dst;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (instr->InputCount() == 2) {
        ASSEMBLE_SIMD_INSTR(Movups, kScratchDoubleReg, 1);
        __ Psrlw(kScratchDoubleReg, byte{8});
        src2 = kScratchDoubleReg;
      }
      __ Psrlw(dst, byte{8});
      __ Packuswb(dst, src2);
      break;
    }
    case kX64S8x16UnzipLow: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src2 = dst;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (instr->InputCount() == 2) {
        ASSEMBLE_SIMD_INSTR(Movups, kScratchDoubleReg, 1);
        __ Psllw(kScratchDoubleReg, byte{8});
        __ Psrlw(kScratchDoubleReg, byte{8});
        src2 = kScratchDoubleReg;
      }
      __ Psllw(dst, byte{8});
      __ Psrlw(dst, byte{8});
      __ Packuswb(dst, src2);
      break;
    }
    case kX64S8x16TransposeLow: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ Psllw(dst, byte{8});
      if (instr->InputCount() == 1) {
        __ Movups(kScratchDoubleReg, dst);
      } else {
        DCHECK_EQ(2, instr->InputCount());
        ASSEMBLE_SIMD_INSTR(Movups, kScratchDoubleReg, 1);
        __ Psllw(kScratchDoubleReg, byte{8});
      }
      __ Psrlw(dst, byte{8});
      __ Por(dst, kScratchDoubleReg);
      break;
    }
    case kX64S8x16TransposeHigh: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ Psrlw(dst, byte{8});
      if (instr->InputCount() == 1) {
        __ Movups(kScratchDoubleReg, dst);
      } else {
        DCHECK_EQ(2, instr->InputCount());
        ASSEMBLE_SIMD_INSTR(Movups, kScratchDoubleReg, 1);
        __ Psrlw(kScratchDoubleReg, byte{8});
      }
      __ Psllw(kScratchDoubleReg, byte{8});
      __ Por(dst, kScratchDoubleReg);
      break;
    }
    case kX64S8x8Reverse:
    case kX64S8x4Reverse:
    case kX64S8x2Reverse: {
      DCHECK_EQ(1, instr->InputCount());
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (arch_opcode != kX64S8x2Reverse) {
        // First shuffle words into position.
        uint8_t shuffle_mask = arch_opcode == kX64S8x4Reverse ? 0xB1 : 0x1B;
        __ Pshuflw(dst, dst, shuffle_mask);
        __ Pshufhw(dst, dst, shuffle_mask);
      }
      __ Movaps(kScratchDoubleReg, dst);
      __ Psrlw(kScratchDoubleReg, byte{8});
      __ Psllw(dst, byte{8});
      __ Por(dst, kScratchDoubleReg);
      break;
    }
    case kX64V32x4AnyTrue:
    case kX64V16x8AnyTrue:
    case kX64V8x16AnyTrue: {
      Register dst = i.OutputRegister();
      XMMRegister src = i.InputSimd128Register(0);

      __ xorq(dst, dst);
      __ Ptest(src, src);
      __ setcc(not_equal, dst);
      break;
    }
    // Need to split up all the different lane structures because the
    // comparison instruction used matters, e.g. given 0xff00, pcmpeqb returns
    // 0x0011, pcmpeqw returns 0x0000, ptest will set ZF to 0 and 1
    // respectively.
    case kX64V32x4AllTrue: {
      ASSEMBLE_SIMD_ALL_TRUE(Pcmpeqd);
      break;
    }
    case kX64V16x8AllTrue: {
      ASSEMBLE_SIMD_ALL_TRUE(Pcmpeqw);
      break;
    }
    case kX64V8x16AllTrue: {
      ASSEMBLE_SIMD_ALL_TRUE(Pcmpeqb);
      break;
    }
    case kWord32AtomicExchangeInt8: {
      __ xchgb(i.InputRegister(0), i.MemoryOperand(1));
      __ movsxbl(i.InputRegister(0), i.InputRegister(0));
      break;
    }
    case kWord32AtomicExchangeUint8: {
      __ xchgb(i.InputRegister(0), i.MemoryOperand(1));
      __ movzxbl(i.InputRegister(0), i.InputRegister(0));
      break;
    }
    case kWord32AtomicExchangeInt16: {
      __ xchgw(i.InputRegister(0), i.MemoryOperand(1));
      __ movsxwl(i.InputRegister(0), i.InputRegister(0));
      break;
    }
    case kWord32AtomicExchangeUint16: {
      __ xchgw(i.InputRegister(0), i.MemoryOperand(1));
      __ movzxwl(i.InputRegister(0), i.InputRegister(0));
      break;
    }
    case kWord32AtomicExchangeWord32: {
      __ xchgl(i.InputRegister(0), i.MemoryOperand(1));
      break;
    }
    case kWord32AtomicCompareExchangeInt8: {
      __ lock();
      __ cmpxchgb(i.MemoryOperand(2), i.InputRegister(1));
      __ movsxbl(rax, rax);
      break;
    }
    case kWord32AtomicCompareExchangeUint8: {
      __ lock();
      __ cmpxchgb(i.MemoryOperand(2), i.InputRegister(1));
      __ movzxbl(rax, rax);
      break;
    }
    case kWord32AtomicCompareExchangeInt16: {
      __ lock();
      __ cmpxchgw(i.MemoryOperand(2), i.InputRegister(1));
      __ movsxwl(rax, rax);
      break;
    }
    case kWord32AtomicCompareExchangeUint16: {
      __ lock();
      __ cmpxchgw(i.MemoryOperand(2), i.InputRegister(1));
      __ movzxwl(rax, rax);
      break;
    }
    case kWord32AtomicCompareExchangeWord32: {
      __ lock();
      __ cmpxchgl(i.MemoryOperand(2), i.InputRegister(1));
      break;
    }
#define ATOMIC_BINOP_CASE(op, inst)              \
  case kWord32Atomic##op##Int8:                  \
    ASSEMBLE_ATOMIC_BINOP(inst, movb, cmpxchgb); \
    __ movsxbl(rax, rax);                        \
    break;                                       \
  case kWord32Atomic##op##Uint8:                 \
    ASSEMBLE_ATOMIC_BINOP(inst, movb, cmpxchgb); \
    __ movzxbl(rax, rax);                        \
    break;                                       \
  case kWord32Atomic##op##Int16:                 \
    ASSEMBLE_ATOMIC_BINOP(inst, movw, cmpxchgw); \
    __ movsxwl(rax, rax);                        \
    break;                                       \
  case kWord32Atomic##op##Uint16:                \
    ASSEMBLE_ATOMIC_BINOP(inst, movw, cmpxchgw); \
    __ movzxwl(rax, rax);                        \
    break;                                       \
  case kWord32Atomic##op##Word32:                \
    ASSEMBLE_ATOMIC_BINOP(inst, movl, cmpxchgl); \
    break;
      ATOMIC_BINOP_CASE(Add, addl)
      ATOMIC_BINOP_CASE(Sub, subl)
      ATOMIC_BINOP_CASE(And, andl)
      ATOMIC_BINOP_CASE(Or, orl)
      ATOMIC_BINOP_CASE(Xor, xorl)
#undef ATOMIC_BINOP_CASE
    case kX64Word64AtomicExchangeUint8: {
      __ xchgb(i.InputRegister(0), i.MemoryOperand(1));
      __ movzxbq(i.InputRegister(0), i.InputRegister(0));
      break;
    }
    case kX64Word64AtomicExchangeUint16: {
      __ xchgw(i.InputRegister(0), i.MemoryOperand(1));
      __ movzxwq(i.InputRegister(0), i.InputRegister(0));
      break;
    }
    case kX64Word64AtomicExchangeUint32: {
      __ xchgl(i.InputRegister(0), i.MemoryOperand(1));
      break;
    }
    case kX64Word64AtomicExchangeUint64: {
      __ xchgq(i.InputRegister(0), i.MemoryOperand(1));
      break;
    }
    case kX64Word64AtomicCompareExchangeUint8: {
      __ lock();
      __ cmpxchgb(i.MemoryOperand(2), i.InputRegister(1));
      __ movzxbq(rax, rax);
      break;
    }
    case kX64Word64AtomicCompareExchangeUint16: {
      __ lock();
      __ cmpxchgw(i.MemoryOperand(2), i.InputRegister(1));
      __ movzxwq(rax, rax);
      break;
    }
    case kX64Word64AtomicCompareExchangeUint32: {
      __ lock();
      __ cmpxchgl(i.MemoryOperand(2), i.InputRegister(1));
      // Zero-extend the 32 bit value to 64 bit.
      __ movl(rax, rax);
      break;
    }
    case kX64Word64AtomicCompareExchangeUint64: {
      __ lock();
      __ cmpxchgq(i.MemoryOperand(2), i.InputRegister(1));
      break;
    }
#define ATOMIC64_BINOP_CASE(op, inst)              \
  case kX64Word64Atomic##op##Uint8:                \
    ASSEMBLE_ATOMIC64_BINOP(inst, movb, cmpxchgb); \
    __ movzxbq(rax, rax);                          \
    break;                                         \
  case kX64Word64Atomic##op##Uint16:               \
    ASSEMBLE_ATOMIC64_BINOP(inst, movw, cmpxchgw); \
    __ movzxwq(rax, rax);                          \
    break;                                         \
  case kX64Word64Atomic##op##Uint32:               \
    ASSEMBLE_ATOMIC64_BINOP(inst, movl, cmpxchgl); \
    break;                                         \
  case kX64Word64Atomic##op##Uint64:               \
    ASSEMBLE_ATOMIC64_BINOP(inst, movq, cmpxchgq); \
    break;
      ATOMIC64_BINOP_CASE(Add, addq)
      ATOMIC64_BINOP_CASE(Sub, subq)
      ATOMIC64_BINOP_CASE(And, andq)
      ATOMIC64_BINOP_CASE(Or, orq)
      ATOMIC64_BINOP_CASE(Xor, xorq)
#undef ATOMIC64_BINOP_CASE
    case kWord32AtomicLoadInt8:
    case kWord32AtomicLoadUint8:
    case kWord32AtomicLoadInt16:
    case kWord32AtomicLoadUint16:
    case kWord32AtomicLoadWord32:
    case kWord32AtomicStoreWord8:
    case kWord32AtomicStoreWord16:
    case kWord32AtomicStoreWord32:
    case kX64Word64AtomicLoadUint8:
    case kX64Word64AtomicLoadUint16:
    case kX64Word64AtomicLoadUint32:
    case kX64Word64AtomicLoadUint64:
    case kX64Word64AtomicStoreWord8:
    case kX64Word64AtomicStoreWord16:
    case kX64Word64AtomicStoreWord32:
    case kX64Word64AtomicStoreWord64:
      UNREACHABLE();  // Won't be generated by instruction selector.
      break;
  }
  return kSuccess;
}  // NOLadability/fn_size)
