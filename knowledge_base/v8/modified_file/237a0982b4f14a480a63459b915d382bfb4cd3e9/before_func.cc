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
            HasCallDescriptorFlag(instr, CallDescriptor::kFixedTargetRegister),
            reg == kJavaScriptCallCodeStartRegister);
        __ LoadCodeObjectEntry(reg, reg);
        if (HasCallDescriptorFlag(instr, CallDescriptor::kRetpoline)) {
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
          if (HasCallDescriptorFlag(instr, CallDescriptor::kRetpoline)) {
            __ RetpolineCall(wasm_code, constant.rmode());
          } else {
            __ Call(wasm_code, constant.rmode());
          }
        }
      } else {
        Register reg = i.InputRegister(0);
        if (HasCallDescriptorFlag(instr, CallDescriptor::kRetpoline)) {
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
    case kArchTailCallCodeObject: {
      if (arch_opcode == kArchTailCallCodeObjectFromJSFunction) {
        AssemblePopArgumentsAdaptorFrame(kJavaScriptCallArgCountRegister,
                                         i.TempRegister(0), i.TempRegister(1),
                                         i.TempRegister(2));
      }
      if (HasImmediateInput(instr, 0)) {
        Handle<Code> code = i.InputCode(0);
        __ Jump(code, RelocInfo::CODE_TARGET);
      } else {
        Register reg = i.InputRegister(0);
        DCHECK_IMPLIES(
            HasCallDescriptorFlag(instr, CallDescriptor::kFixedTargetRegister),
            reg == kJavaScriptCallCodeStartRegister);
        __ LoadCodeObjectEntry(reg, reg);
        if (HasCallDescriptorFlag(instr, CallDescriptor::kRetpoline)) {
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
        if (HasCallDescriptorFlag(instr, CallDescriptor::kRetpoline)) {
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
          HasCallDescriptorFlag(instr, CallDescriptor::kFixedTargetRegister),
          reg == kJavaScriptCallCodeStartRegister);
      if (HasCallDescriptorFlag(instr, CallDescriptor::kRetpoline)) {
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
      CodeGenResult result = AssembleDeoptimizerCall(exit);
      if (result != kSuccess) return result;
      unwinding_info_writer_.MarkBlockWillExit();
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
      auto ool = new (zone()) OutOfLineTruncateDoubleToI(
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
      auto ool = new (zone())
          OutOfLineRecordWrite(this, object, operand, value, scratch0, scratch1,
                               mode, DetermineStubCallMode());
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
          new (zone()) OutOfLineLoadFloat32NaN(this, i.OutputDoubleRegister());
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
          new (zone()) OutOfLineLoadFloat32NaN(this, i.OutputDoubleRegister());
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
          new (zone()) OutOfLineLoadFloat64NaN(this, i.OutputDoubleRegister());
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
          new (zone()) OutOfLineLoadFloat64NaN(this, i.OutputDoubleRegister());
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
        } else {
          DCHECK_EQ(MachineRepresentation::kFloat32, op->representation());
          __ Movss(i.OutputFloatRegister(), Operand(rbp, offset));
        }
      } else {
        __ movq(i.OutputRegister(), Operand(rbp, offset));
      }
      break;
    }
    case kX64F64x2Splat: {
      CpuFeatureScope sse_scope(tasm(), SSE3);
      XMMRegister dst = i.OutputSimd128Register();
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Movddup(dst, i.InputDoubleRegister(0));
      } else {
        __ Movddup(dst, i.InputOperand(0));
      }
      break;
    }
    case kX64F64x2ReplaceLane: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      if (instr->InputAt(2)->IsFPRegister()) {
        __ Movq(kScratchRegister, i.InputDoubleRegister(2));
        __ Pinsrq(i.OutputSimd128Register(), kScratchRegister, i.InputInt8(1));
      } else {
        __ Pinsrq(i.OutputSimd128Register(), i.InputOperand(2), i.InputInt8(1));
      }
      break;
    }
    case kX64F64x2ExtractLane: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pextrq(kScratchRegister, i.InputSimd128Register(0), i.InputInt8(1));
      __ Movq(i.OutputDoubleRegister(), kScratchRegister);
      break;
    }
    case kX64F64x2Sqrt: {
      __ Sqrtpd(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64F64x2Add: {
      ASSEMBLE_SSE_BINOP(Addpd);
      break;
    }
    case kX64F64x2Sub: {
      ASSEMBLE_SSE_BINOP(Subpd);
      break;
    }
    case kX64F64x2Mul: {
      ASSEMBLE_SSE_BINOP(Mulpd);
      break;
    }
    case kX64F64x2Div: {
      ASSEMBLE_SSE_BINOP(Divpd);
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
      __ Cmppd(dst, kScratchDoubleReg, static_cast<int8_t>(3));
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
      __ Cmppd(dst, kScratchDoubleReg, static_cast<int8_t>(3));
      __ Psrlq(dst, 13);
      __ Andnpd(dst, kScratchDoubleReg);
      break;
    }
    case kX64F64x2Eq: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Cmpeqpd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F64x2Ne: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Cmpneqpd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F64x2Lt: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Cmpltpd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F64x2Le: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Cmplepd(i.OutputSimd128Register(), i.InputSimd128Register(1));
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
    // TODO(gdeepti): Get rid of redundant moves for F32x4Splat/Extract below
    case kX64F32x4Splat: {
      XMMRegister dst = i.OutputSimd128Register();
      if (instr->InputAt(0)->IsFPRegister()) {
        __ Movss(dst, i.InputDoubleRegister(0));
      } else {
        __ Movss(dst, i.InputOperand(0));
      }
      __ Shufps(dst, dst, static_cast<byte>(0x0));
      break;
    }
    case kX64F32x4ExtractLane: {
      __ Extractps(kScratchRegister, i.InputSimd128Register(0), i.InputInt8(1));
      __ Movd(i.OutputDoubleRegister(), kScratchRegister);
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
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      __ Pxor(kScratchDoubleReg, kScratchDoubleReg);  // zeros
      __ Pblendw(kScratchDoubleReg, dst,
                 static_cast<uint8_t>(0x55));             // get lo 16 bits
      __ Psubd(dst, kScratchDoubleReg);                   // get hi 16 bits
      __ Cvtdq2ps(kScratchDoubleReg, kScratchDoubleReg);  // convert lo exactly
      __ Psrld(dst,
               static_cast<byte>(1));    // divide by 2 to get in unsigned range
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
        __ Psrld(kScratchDoubleReg, static_cast<byte>(1));
        __ Andps(i.OutputSimd128Register(), kScratchDoubleReg);
      } else {
        __ Pcmpeqd(dst, dst);
        __ Psrld(dst, static_cast<byte>(1));
        __ Andps(dst, i.InputSimd128Register(0));
      }
      break;
    }
    case kX64F32x4Neg: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ Pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
        __ Pslld(kScratchDoubleReg, static_cast<byte>(31));
        __ Xorps(i.OutputSimd128Register(), kScratchDoubleReg);
      } else {
        __ Pcmpeqd(dst, dst);
        __ Pslld(dst, static_cast<byte>(31));
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
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Addps(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F32x4AddHoriz: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      CpuFeatureScope sse_scope(tasm(), SSE3);
      __ Haddps(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F32x4Sub: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Subps(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F32x4Mul: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Mulps(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F32x4Div: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Divps(i.OutputSimd128Register(), i.InputSimd128Register(1));
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
      __ Cmpps(dst, kScratchDoubleReg, static_cast<int8_t>(3));
      __ Orps(kScratchDoubleReg, dst);
      __ Psrld(dst, static_cast<byte>(10));
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
      __ Cmpps(dst, kScratchDoubleReg, static_cast<int8_t>(3));
      __ Psrld(dst, static_cast<byte>(10));
      __ Andnps(dst, kScratchDoubleReg);
      break;
    }
    case kX64F32x4Eq: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Cmpps(i.OutputSimd128Register(), i.InputSimd128Register(1),
               static_cast<int8_t>(0x0));
      break;
    }
    case kX64F32x4Ne: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Cmpps(i.OutputSimd128Register(), i.InputSimd128Register(1),
               static_cast<int8_t>(0x4));
      break;
    }
    case kX64F32x4Lt: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Cmpltps(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64F32x4Le: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Cmpleps(i.OutputSimd128Register(), i.InputSimd128Register(1));
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
    case kX64I64x2Splat: {
      CpuFeatureScope sse_scope(tasm(), SSE3);
      XMMRegister dst = i.OutputSimd128Register();
      if (HasRegisterInput(instr, 0)) {
        __ Movq(dst, i.InputRegister(0));
      } else {
        __ Movq(dst, i.InputOperand(0));
      }
      __ Movddup(dst, dst);
      break;
    }
    case kX64I64x2ExtractLane: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pextrq(i.OutputRegister(), i.InputSimd128Register(0), i.InputInt8(1));
      break;
    }
    case kX64I64x2ReplaceLane: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      if (HasRegisterInput(instr, 2)) {
        __ Pinsrq(i.OutputSimd128Register(), i.InputRegister(2),
                  i.InputInt8(1));
      } else {
        __ Pinsrq(i.OutputSimd128Register(), i.InputOperand(2), i.InputInt8(1));
      }
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
    case kX64I64x2Shl: {
      // Take shift value modulo 2^6.
      ASSEMBLE_SIMD_SHIFT(Psllq, 6);
      break;
    }
    case kX64I64x2ShrS: {
      // TODO(zhin): there is vpsraq but requires AVX512
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      // ShrS on each quadword one at a time
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      Register tmp = i.ToRegister(instr->TempAt(0));
      // Modulo 64 not required as sarq_cl will mask cl to 6 bits.

      // lower quadword
      __ Pextrq(tmp, src, static_cast<int8_t>(0x0));
      __ sarq_cl(tmp);
      __ Pinsrq(dst, tmp, static_cast<int8_t>(0x0));

      // upper quadword
      __ Pextrq(tmp, src, static_cast<int8_t>(0x1));
      __ sarq_cl(tmp);
      __ Pinsrq(dst, tmp, static_cast<int8_t>(0x1));
      break;
    }
    case kX64I64x2Add: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Paddq(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I64x2Sub: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Psubq(i.OutputSimd128Register(), i.InputSimd128Register(1));
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
    case kX64I64x2MinS: {
      if (CpuFeatures::IsSupported(SSE4_2)) {
        CpuFeatureScope sse_scope_4_2(tasm(), SSE4_2);
        XMMRegister dst = i.OutputSimd128Register();
        XMMRegister src0 = i.InputSimd128Register(0);
        XMMRegister src1 = i.InputSimd128Register(1);
        XMMRegister tmp = i.TempSimd128Register(0);
        DCHECK_EQ(tmp, xmm0);

        __ movaps(tmp, src1);
        __ pcmpgtq(tmp, src0);
        __ movaps(dst, src1);
        __ blendvpd(dst, src0);  // implicit use of xmm0 as mask
      } else {
        CpuFeatureScope sse_scope_4_1(tasm(), SSE4_1);
        XMMRegister dst = i.OutputSimd128Register();
        XMMRegister src = i.InputSimd128Register(1);
        XMMRegister tmp = i.TempSimd128Register(0);
        Register tmp1 = i.TempRegister(1);
        Register tmp2 = i.TempRegister(2);
        DCHECK_EQ(dst, i.InputSimd128Register(0));
        // backup src since we cannot change it
        __ movaps(tmp, src);

        // compare the lower quardwords
        __ movq(tmp1, dst);
        __ movq(tmp2, tmp);
        __ cmpq(tmp1, tmp2);
        // tmp2 now has the min of lower quadwords
        __ cmovq(less_equal, tmp2, tmp1);
        // tmp1 now has the higher quadword
        // must do this before movq, movq clears top quadword
        __ pextrq(tmp1, dst, 1);
        // save tmp2 into dst
        __ movq(dst, tmp2);
        // tmp2 now has the higher quadword
        __ pextrq(tmp2, tmp, 1);
        //  compare higher quadwords
        __ cmpq(tmp1, tmp2);
        // tmp2 now has the min of higher quadwords
        __ cmovq(less_equal, tmp2, tmp1);
        __ movq(tmp, tmp2);
        // dst = [tmp[0], dst[0]]
        __ punpcklqdq(dst, tmp);
      }
      break;
    }
    case kX64I64x2MaxS: {
      CpuFeatureScope sse_scope_4_2(tasm(), SSE4_2);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      DCHECK_EQ(tmp, xmm0);

      __ movaps(tmp, src);
      __ pcmpgtq(tmp, dst);
      __ blendvpd(dst, src);  // implicit use of xmm0 as mask
      break;
    }
    case kX64I64x2Eq: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Pcmpeqq(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I64x2Ne: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister tmp = i.TempSimd128Register(0);
      __ Pcmpeqq(i.OutputSimd128Register(), i.InputSimd128Register(1));
      __ Pcmpeqq(tmp, tmp);
      __ Pxor(i.OutputSimd128Register(), tmp);
      break;
    }
    case kX64I64x2GtS: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ Pcmpgtq(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I64x2GeS: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);

      __ Movaps(tmp, src);
      __ Pcmpgtq(tmp, dst);
      __ Pcmpeqd(dst, dst);
      __ Pxor(dst, tmp);
      break;
    }
    case kX64I64x2ShrU: {
      // Take shift value modulo 2^6.
      ASSEMBLE_SIMD_SHIFT(Psrlq, 6);
      break;
    }
    case kX64I64x2MinU: {
      CpuFeatureScope sse_scope_4_2(tasm(), SSE4_2);
      CpuFeatureScope sse_scope_4_1(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src0 = i.InputSimd128Register(0);
      XMMRegister src1 = i.InputSimd128Register(1);
      XMMRegister tmp0 = i.TempSimd128Register(0);
      XMMRegister tmp1 = i.TempSimd128Register(1);
      DCHECK_EQ(tmp1, xmm0);

      __ movaps(dst, src1);
      __ movaps(tmp0, src0);

      __ pcmpeqd(tmp1, tmp1);
      __ psllq(tmp1, 63);

      __ pxor(tmp0, tmp1);
      __ pxor(tmp1, dst);

      __ pcmpgtq(tmp1, tmp0);
      __ blendvpd(dst, src0);  // implicit use of xmm0 as mask
      break;
    }
    case kX64I64x2MaxU: {
      CpuFeatureScope sse_scope_4_2(tasm(), SSE4_2);
      CpuFeatureScope sse_scope_4_1(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister dst_tmp = i.TempSimd128Register(0);
      XMMRegister tmp = i.TempSimd128Register(1);
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      DCHECK_EQ(tmp, xmm0);

      __ movaps(dst_tmp, dst);

      __ pcmpeqd(tmp, tmp);
      __ psllq(tmp, 63);

      __ pxor(dst_tmp, tmp);
      __ pxor(tmp, src);

      __ pcmpgtq(tmp, dst_tmp);
      __ blendvpd(dst, src);  // implicit use of xmm0 as mask
      break;
    }
    case kX64I64x2GtU: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);

      __ Pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
      __ Psllq(kScratchDoubleReg, 63);

      __ Movaps(tmp, src);
      __ Pxor(tmp, kScratchDoubleReg);
      __ Pxor(dst, kScratchDoubleReg);
      __ Pcmpgtq(dst, tmp);
      break;
    }
    case kX64I64x2GeU: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      CpuFeatureScope sse_scope(tasm(), SSE4_2);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);

      __ Pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
      __ Psllq(kScratchDoubleReg, 63);

      __ Movaps(tmp, src);
      __ Pxor(dst, kScratchDoubleReg);
      __ Pxor(tmp, kScratchDoubleReg);
      __ Pcmpgtq(tmp, dst);
      __ Pcmpeqd(dst, dst);
      __ Pxor(dst, tmp);
      break;
    }
    case kX64I32x4Splat: {
      XMMRegister dst = i.OutputSimd128Register();
      if (HasRegisterInput(instr, 0)) {
        __ Movd(dst, i.InputRegister(0));
      } else {
        __ Movd(dst, i.InputOperand(0));
      }
      __ Pshufd(dst, dst, static_cast<uint8_t>(0x0));
      break;
    }
    case kX64I32x4ExtractLane: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pextrd(i.OutputRegister(), i.InputSimd128Register(0), i.InputInt8(1));
      break;
    }
    case kX64I32x4ReplaceLane: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      if (HasRegisterInput(instr, 2)) {
        __ Pinsrd(i.OutputSimd128Register(), i.InputRegister(2),
                  i.InputInt8(1));
      } else {
        __ Pinsrd(i.OutputSimd128Register(), i.InputOperand(2), i.InputInt8(1));
      }
      break;
    }
    case kX64I32x4SConvertF32x4: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister tmp = i.TempSimd128Register(0);
      // NAN->0
      __ movaps(tmp, dst);
      __ cmpeqps(tmp, tmp);
      __ pand(dst, tmp);
      // Set top bit if >= 0 (but not -0.0!)
      __ pxor(tmp, dst);
      // Convert
      __ cvttps2dq(dst, dst);
      // Set top bit if >=0 is now < 0
      __ pand(tmp, dst);
      __ psrad(tmp, 31);
      // Set positive overflow lanes to 0x7FFFFFFF
      __ pxor(dst, tmp);
      break;
    }
    case kX64I32x4SConvertI16x8Low: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pmovsxwd(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I32x4SConvertI16x8High: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Palignr(dst, i.InputSimd128Register(0), static_cast<uint8_t>(8));
      __ Pmovsxwd(dst, dst);
      break;
    }
    case kX64I32x4Neg: {
      CpuFeatureScope sse_scope(tasm(), SSSE3);
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
      ASSEMBLE_SIMD_SHIFT(Pslld, 5);
      break;
    }
    case kX64I32x4ShrS: {
      // Take shift value modulo 2^5.
      ASSEMBLE_SIMD_SHIFT(Psrad, 5);
      break;
    }
    case kX64I32x4Add: {
      __ Paddd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4AddHoriz: {
      CpuFeatureScope sse_scope(tasm(), SSSE3);
      __ phaddd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4Sub: {
      __ Psubd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4Mul: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pmulld(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4MinS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pminsd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4MaxS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pmaxsd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4Eq: {
      __ Pcmpeqd(i.OutputSimd128Register(), i.InputSimd128Register(1));
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
      __ Pcmpgtd(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4GeS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ Pminsd(dst, src);
      __ Pcmpeqd(dst, src);
      break;
    }
    case kX64I32x4UConvertF32x4: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister tmp = i.TempSimd128Register(0);
      XMMRegister tmp2 = i.TempSimd128Register(1);
      // NAN->0, negative->0
      __ pxor(tmp2, tmp2);
      __ maxps(dst, tmp2);
      // scratch: float representation of max_signed
      __ pcmpeqd(tmp2, tmp2);
      __ psrld(tmp2, 1);        // 0x7fffffff
      __ cvtdq2ps(tmp2, tmp2);  // 0x4f000000
      // tmp: convert (src-max_signed).
      // Positive overflow lanes -> 0x7FFFFFFF
      // Negative lanes -> 0
      __ movaps(tmp, dst);
      __ subps(tmp, tmp2);
      __ cmpleps(tmp2, tmp);
      __ cvttps2dq(tmp, tmp);
      __ pxor(tmp, tmp2);
      __ pxor(tmp2, tmp2);
      __ pmaxsd(tmp, tmp2);
      // convert. Overflow lanes above max_signed will be 0x80000000
      __ cvttps2dq(dst, dst);
      // Add (src-max_signed) for overflow lanes.
      __ paddd(dst, tmp);
      break;
    }
    case kX64I32x4UConvertI16x8Low: {
      __ Pmovzxwd(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I32x4UConvertI16x8High: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Palignr(dst, i.InputSimd128Register(0), static_cast<uint8_t>(8));
      __ Pmovzxwd(dst, dst);
      break;
    }
    case kX64I32x4ShrU: {
      // Take shift value modulo 2^5.
      ASSEMBLE_SIMD_SHIFT(Psrld, 5);
      break;
    }
    case kX64I32x4MinU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pminud(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4MaxU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ Pmaxud(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I32x4GtU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
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
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
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
    case kX64S128Zero: {
      XMMRegister dst = i.OutputSimd128Register();
      __ xorps(dst, dst);
      break;
    }
    case kX64I16x8Splat: {
      XMMRegister dst = i.OutputSimd128Register();
      if (HasRegisterInput(instr, 0)) {
        __ movd(dst, i.InputRegister(0));
      } else {
        __ movd(dst, i.InputOperand(0));
      }
      __ pshuflw(dst, dst, 0x0);
      __ pshufd(dst, dst, 0x0);
      break;
    }
    case kX64I16x8ExtractLaneU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      Register dst = i.OutputRegister();
      __ Pextrw(dst, i.InputSimd128Register(0), i.InputInt8(1));
      break;
    }
    case kX64I16x8ExtractLaneS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      Register dst = i.OutputRegister();
      __ Pextrw(dst, i.InputSimd128Register(0), i.InputInt8(1));
      __ movsxwl(dst, dst);
      break;
    }
    case kX64I16x8ReplaceLane: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      if (HasRegisterInput(instr, 2)) {
        __ Pinsrw(i.OutputSimd128Register(), i.InputRegister(2),
                  i.InputInt8(1));
      } else {
        __ Pinsrw(i.OutputSimd128Register(), i.InputOperand(2), i.InputInt8(1));
      }
      break;
    }
    case kX64I16x8SConvertI8x16Low: {
      __ Pmovsxbw(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I16x8SConvertI8x16High: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Palignr(dst, i.InputSimd128Register(0), static_cast<uint8_t>(8));
      __ Pmovsxbw(dst, dst);
      break;
    }
    case kX64I16x8Neg: {
      CpuFeatureScope sse_scope(tasm(), SSSE3);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
        __ psignw(dst, kScratchDoubleReg);
      } else {
        __ pxor(dst, dst);
        __ psubw(dst, src);
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
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ packssdw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8Add: {
      __ paddw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8AddSaturateS: {
      __ paddsw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8AddHoriz: {
      CpuFeatureScope sse_scope(tasm(), SSSE3);
      __ phaddw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8Sub: {
      __ psubw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8SubSaturateS: {
      __ psubsw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8Mul: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pmullw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8MinS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pminsw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8MaxS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pmaxsw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8Eq: {
      __ pcmpeqw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8Ne: {
      XMMRegister tmp = i.TempSimd128Register(0);
      __ pcmpeqw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      __ pcmpeqw(tmp, tmp);
      __ pxor(i.OutputSimd128Register(), tmp);
      break;
    }
    case kX64I16x8GtS: {
      __ pcmpgtw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8GeS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ pminsw(dst, src);
      __ pcmpeqw(dst, src);
      break;
    }
    case kX64I16x8UConvertI8x16Low: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pmovzxbw(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I16x8UConvertI8x16High: {
      XMMRegister dst = i.OutputSimd128Register();
      __ Palignr(dst, i.InputSimd128Register(0), static_cast<uint8_t>(8));
      __ Pmovzxbw(dst, dst);
      break;
    }
    case kX64I16x8ShrU: {
      // Take shift value modulo 2^4.
      ASSEMBLE_SIMD_SHIFT(psrlw, 4);
      break;
    }
    case kX64I16x8UConvertI32x4: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ packusdw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8AddSaturateU: {
      __ paddusw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8SubSaturateU: {
      __ psubusw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8MinU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pminuw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8MaxU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pmaxuw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8GtU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);
      __ pmaxuw(dst, src);
      __ pcmpeqw(dst, src);
      __ pcmpeqw(tmp, tmp);
      __ pxor(dst, tmp);
      break;
    }
    case kX64I16x8GeU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ pminuw(dst, src);
      __ pcmpeqw(dst, src);
      break;
    }
    case kX64I16x8RoundingAverageU: {
      __ Pavgw(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I16x8Abs: {
      __ Pabsw(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64I8x16Splat: {
      CpuFeatureScope sse_scope(tasm(), SSSE3);
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
    case kX64I8x16ExtractLaneU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      Register dst = i.OutputRegister();
      __ Pextrb(dst, i.InputSimd128Register(0), i.InputInt8(1));
      break;
    }
    case kX64I8x16ExtractLaneS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      Register dst = i.OutputRegister();
      __ Pextrb(dst, i.InputSimd128Register(0), i.InputInt8(1));
      __ movsxbl(dst, dst);
      break;
    }
    case kX64I8x16ReplaceLane: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      if (HasRegisterInput(instr, 2)) {
        __ Pinsrb(i.OutputSimd128Register(), i.InputRegister(2),
                  i.InputInt8(1));
      } else {
        __ Pinsrb(i.OutputSimd128Register(), i.InputOperand(2), i.InputInt8(1));
      }
      break;
    }
    case kX64I8x16SConvertI16x8: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      __ packsswb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16Neg: {
      CpuFeatureScope sse_scope(tasm(), SSSE3);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ pcmpeqd(kScratchDoubleReg, kScratchDoubleReg);
        __ psignb(dst, kScratchDoubleReg);
      } else {
        __ pxor(dst, dst);
        __ psubb(dst, src);
      }
      break;
    }
    case kX64I8x16Shl: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // Temp registers for shift mask andadditional moves to XMM registers.
      Register tmp = i.ToRegister(instr->TempAt(0));
      XMMRegister tmp_simd = i.TempSimd128Register(1);
      Register shift = i.InputRegister(1);
      // Mask off the unwanted bits before word-shifting.
      __ pcmpeqw(kScratchDoubleReg, kScratchDoubleReg);
      // Take shift value modulo 8.
      __ andq(shift, Immediate(7));
      __ movq(tmp, shift);
      __ addq(tmp, Immediate(8));
      __ movq(tmp_simd, tmp);
      __ psrlw(kScratchDoubleReg, tmp_simd);
      __ packuswb(kScratchDoubleReg, kScratchDoubleReg);
      __ pand(dst, kScratchDoubleReg);
      __ movq(tmp_simd, shift);
      __ psllw(dst, tmp_simd);
      break;
    }
    case kX64I8x16ShrS: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // Temp registers for shift mask andadditional moves to XMM registers.
      Register tmp = i.ToRegister(instr->TempAt(0));
      XMMRegister tmp_simd = i.TempSimd128Register(1);
      // Unpack the bytes into words, do arithmetic shifts, and repack.
      __ punpckhbw(kScratchDoubleReg, dst);
      __ punpcklbw(dst, dst);
      // Prepare shift value
      __ movq(tmp, i.InputRegister(1));
      // Take shift value modulo 8.
      __ andq(tmp, Immediate(7));
      __ addq(tmp, Immediate(8));
      __ movq(tmp_simd, tmp);
      __ psraw(kScratchDoubleReg, tmp_simd);
      __ psraw(dst, tmp_simd);
      __ packsswb(dst, kScratchDoubleReg);
      break;
    }
    case kX64I8x16Add: {
      __ paddb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16AddSaturateS: {
      __ paddsb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16Sub: {
      __ psubb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16SubSaturateS: {
      __ psubsb(i.OutputSimd128Register(), i.InputSimd128Register(1));
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
      __ movaps(tmp, dst);
      __ movaps(kScratchDoubleReg, right);
      __ psrlw(tmp, 8);
      __ psrlw(kScratchDoubleReg, 8);
      // dst = left * 256
      __ psllw(dst, 8);
      // t = I16x8Mul(t, s)
      //    => __PP __PP ...  __PP  __PP
      __ pmullw(tmp, kScratchDoubleReg);
      // dst = I16x8Mul(left * 256, right)
      //    => pp__ pp__ ...  pp__  pp__
      __ pmullw(dst, right);
      // t = I16x8Shl(t, 8)
      //    => PP00 PP00 ...  PP00  PP00
      __ psllw(tmp, 8);
      // dst = I16x8Shr(dst, 8)
      //    => 00pp 00pp ...  00pp  00pp
      __ psrlw(dst, 8);
      // dst = I16x8Or(dst, t)
      //    => PPpp PPpp ...  PPpp  PPpp
      __ por(dst, tmp);
      break;
    }
    case kX64I8x16MinS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pminsb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16MaxS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pmaxsb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16Eq: {
      __ pcmpeqb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16Ne: {
      XMMRegister tmp = i.TempSimd128Register(0);
      __ pcmpeqb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      __ pcmpeqb(tmp, tmp);
      __ pxor(i.OutputSimd128Register(), tmp);
      break;
    }
    case kX64I8x16GtS: {
      __ pcmpgtb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16GeS: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ pminsb(dst, src);
      __ pcmpeqb(dst, src);
      break;
    }
    case kX64I8x16UConvertI16x8: {
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ packuswb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16ShrU: {
      XMMRegister dst = i.OutputSimd128Register();
      // Unpack the bytes into words, do logical shifts, and repack.
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      // Temp registers for shift mask andadditional moves to XMM registers.
      Register tmp = i.ToRegister(instr->TempAt(0));
      XMMRegister tmp_simd = i.TempSimd128Register(1);
      __ punpckhbw(kScratchDoubleReg, dst);
      __ punpcklbw(dst, dst);
      // Prepare shift value
      __ movq(tmp, i.InputRegister(1));
      // Take shift value modulo 8.
      __ andq(tmp, Immediate(7));
      __ addq(tmp, Immediate(8));
      __ movq(tmp_simd, tmp);
      __ psrlw(kScratchDoubleReg, tmp_simd);
      __ psrlw(dst, tmp_simd);
      __ packuswb(dst, kScratchDoubleReg);
      break;
    }
    case kX64I8x16AddSaturateU: {
      __ Paddusb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16SubSaturateU: {
      __ psubusb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16MinU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pminub(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16MaxU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      __ pmaxub(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16GtU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      XMMRegister tmp = i.TempSimd128Register(0);
      __ pmaxub(dst, src);
      __ pcmpeqb(dst, src);
      __ pcmpeqb(tmp, tmp);
      __ pxor(dst, tmp);
      break;
    }
    case kX64I8x16GeU: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(1);
      __ pminub(dst, src);
      __ pcmpeqb(dst, src);
      break;
    }
    case kX64I8x16RoundingAverageU: {
      __ Pavgb(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64I8x16Abs: {
      __ Pabsb(i.OutputSimd128Register(), i.InputSimd128Register(0));
      break;
    }
    case kX64S128And: {
      __ pand(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64S128Or: {
      __ por(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64S128Xor: {
      __ pxor(i.OutputSimd128Register(), i.InputSimd128Register(1));
      break;
    }
    case kX64S128Not: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src = i.InputSimd128Register(0);
      if (dst == src) {
        __ movaps(kScratchDoubleReg, dst);
        __ pcmpeqd(dst, dst);
        __ pxor(dst, kScratchDoubleReg);
      } else {
        __ pcmpeqd(dst, dst);
        __ pxor(dst, src);
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
    case kX64S8x16Swizzle: {
      CpuFeatureScope sse_scope(tasm(), SSSE3);
      DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister mask = i.TempSimd128Register(0);

      // Out-of-range indices should return 0, add 112 so that any value > 15
      // saturates to 128 (top bit set), so pshufb will zero that lane.
      __ Move(mask, static_cast<uint32_t>(0x70707070));
      __ Pshufd(mask, mask, static_cast<uint8_t>(0x0));
      __ Paddusb(mask, i.InputSimd128Register(1));
      __ Pshufb(dst, mask);
      break;
    }
    case kX64S8x16Shuffle: {
      XMMRegister dst = i.OutputSimd128Register();
      Register tmp = i.TempRegister(0);
      // Prepare 16 byte aligned buffer for shuffle control mask
      __ movq(tmp, rsp);
      __ andq(rsp, Immediate(-16));
      if (instr->InputCount() == 5) {  // only one input operand
        uint32_t mask[4] = {};
        DCHECK_EQ(i.OutputSimd128Register(), i.InputSimd128Register(0));
        for (int j = 4; j > 0; j--) {
          mask[j - 1] = i.InputUint32(j);
        }

        SetupShuffleMaskOnStack(tasm(), mask);
        __ Pshufb(dst, Operand(rsp, 0));
      } else {  // two input operands
        DCHECK_EQ(6, instr->InputCount());
        ASSEMBLE_SIMD_INSTR(Movups, kScratchDoubleReg, 0);
        uint32_t mask[4] = {};
        for (int j = 5; j > 1; j--) {
          uint32_t lanes = i.InputUint32(j);
          for (int k = 0; k < 32; k += 8) {
            uint8_t lane = lanes >> k;
            mask[j - 2] |= (lane < kSimd128Size ? lane : 0x80) << k;
          }
        }
        SetupShuffleMaskOnStack(tasm(), mask);
        __ Pshufb(kScratchDoubleReg, Operand(rsp, 0));
        uint32_t mask1[4] = {};
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
            mask1[j - 2] |= (lane >= kSimd128Size ? (lane & 0x0F) : 0x80) << k;
          }
        }
        SetupShuffleMaskOnStack(tasm(), mask1);
        __ Pshufb(dst, Operand(rsp, 0));
        __ Por(dst, kScratchDoubleReg);
      }
      __ movq(rsp, tmp);
      break;
    }
    case kX64S8x16LoadSplat: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pinsrb(i.OutputSimd128Register(), i.MemoryOperand(), 0);
      __ Pxor(kScratchDoubleReg, kScratchDoubleReg);
      __ Pshufb(i.OutputSimd128Register(), kScratchDoubleReg);
      break;
    }
    case kX64S16x8LoadSplat: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pinsrw(i.OutputSimd128Register(), i.MemoryOperand(), 0);
      __ Pshuflw(i.OutputSimd128Register(), i.OutputSimd128Register(),
                 static_cast<uint8_t>(0));
      __ Punpcklqdq(i.OutputSimd128Register(), i.OutputSimd128Register());
      break;
    }
    case kX64S32x4LoadSplat: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      if (CpuFeatures::IsSupported(AVX)) {
        CpuFeatureScope avx_scope(tasm(), AVX);
        __ vbroadcastss(i.OutputSimd128Register(), i.MemoryOperand());
      } else {
        __ Movss(i.OutputSimd128Register(), i.MemoryOperand());
        __ Shufps(i.OutputSimd128Register(), i.OutputSimd128Register(),
                  static_cast<byte>(0));
      }
      break;
    }
    case kX64S64x2LoadSplat: {
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Movddup(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64I16x8Load8x8S: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovsxbw(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64I16x8Load8x8U: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovzxbw(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64I32x4Load16x4S: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovsxwd(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64I32x4Load16x4U: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovzxwd(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64I64x2Load32x2S: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovsxdq(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64I64x2Load32x2U: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      EmitOOLTrapIfNeeded(zone(), this, opcode, instr, __ pc_offset());
      __ Pmovzxdq(i.OutputSimd128Register(), i.MemoryOperand());
      break;
    }
    case kX64S32x4Swizzle: {
      DCHECK_EQ(2, instr->InputCount());
      ASSEMBLE_SIMD_IMM_INSTR(pshufd, i.OutputSimd128Register(), 0,
                              i.InputInt8(1));
      break;
    }
    case kX64S32x4Shuffle: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      DCHECK_EQ(4, instr->InputCount());  // Swizzles should be handled above.
      int8_t shuffle = i.InputInt8(2);
      DCHECK_NE(0xe4, shuffle);  // A simple blend should be handled below.
      ASSEMBLE_SIMD_IMM_INSTR(pshufd, kScratchDoubleReg, 1, shuffle);
      ASSEMBLE_SIMD_IMM_INSTR(pshufd, i.OutputSimd128Register(), 0, shuffle);
      __ pblendw(i.OutputSimd128Register(), kScratchDoubleReg, i.InputInt8(3));
      break;
    }
    case kX64S16x8Blend: {
      ASSEMBLE_SIMD_IMM_SHUFFLE(pblendw, SSE4_1, i.InputInt8(2));
      break;
    }
    case kX64S16x8HalfShuffle1: {
      XMMRegister dst = i.OutputSimd128Register();
      ASSEMBLE_SIMD_IMM_INSTR(pshuflw, dst, 0, i.InputInt8(1));
      __ pshufhw(dst, dst, i.InputInt8(2));
      break;
    }
    case kX64S16x8HalfShuffle2: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      ASSEMBLE_SIMD_IMM_INSTR(pshuflw, kScratchDoubleReg, 1, i.InputInt8(2));
      __ pshufhw(kScratchDoubleReg, kScratchDoubleReg, i.InputInt8(3));
      ASSEMBLE_SIMD_IMM_INSTR(pshuflw, dst, 0, i.InputInt8(2));
      __ pshufhw(dst, dst, i.InputInt8(3));
      __ pblendw(dst, kScratchDoubleReg, i.InputInt8(4));
      break;
    }
    case kX64S8x16Alignr: {
      ASSEMBLE_SIMD_IMM_SHUFFLE(palignr, SSSE3, i.InputInt8(2));
      break;
    }
    case kX64S16x8Dup: {
      XMMRegister dst = i.OutputSimd128Register();
      int8_t lane = i.InputInt8(1) & 0x7;
      int8_t lane4 = lane & 0x3;
      int8_t half_dup = lane4 | (lane4 << 2) | (lane4 << 4) | (lane4 << 6);
      if (lane < 4) {
        ASSEMBLE_SIMD_IMM_INSTR(pshuflw, dst, 0, half_dup);
        __ pshufd(dst, dst, 0);
      } else {
        ASSEMBLE_SIMD_IMM_INSTR(pshufhw, dst, 0, half_dup);
        __ pshufd(dst, dst, 0xaa);
      }
      break;
    }
    case kX64S8x16Dup: {
      XMMRegister dst = i.OutputSimd128Register();
      int8_t lane = i.InputInt8(1) & 0xf;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (lane < 8) {
        __ punpcklbw(dst, dst);
      } else {
        __ punpckhbw(dst, dst);
      }
      lane &= 0x7;
      int8_t lane4 = lane & 0x3;
      int8_t half_dup = lane4 | (lane4 << 2) | (lane4 << 4) | (lane4 << 6);
      if (lane < 4) {
        __ pshuflw(dst, dst, half_dup);
        __ pshufd(dst, dst, 0);
      } else {
        __ pshufhw(dst, dst, half_dup);
        __ pshufd(dst, dst, 0xaa);
      }
      break;
    }
    case kX64S64x2UnpackHigh:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(punpckhqdq);
      break;
    case kX64S32x4UnpackHigh:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(punpckhdq);
      break;
    case kX64S16x8UnpackHigh:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(punpckhwd);
      break;
    case kX64S8x16UnpackHigh:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(punpckhbw);
      break;
    case kX64S64x2UnpackLow:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(punpcklqdq);
      break;
    case kX64S32x4UnpackLow:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(punpckldq);
      break;
    case kX64S16x8UnpackLow:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(punpcklwd);
      break;
    case kX64S8x16UnpackLow:
      ASSEMBLE_SIMD_PUNPCK_SHUFFLE(punpcklbw);
      break;
    case kX64S16x8UnzipHigh: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src2 = dst;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (instr->InputCount() == 2) {
        ASSEMBLE_SIMD_INSTR(movups, kScratchDoubleReg, 1);
        __ psrld(kScratchDoubleReg, 16);
        src2 = kScratchDoubleReg;
      }
      __ psrld(dst, 16);
      __ packusdw(dst, src2);
      break;
    }
    case kX64S16x8UnzipLow: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src2 = dst;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ pxor(kScratchDoubleReg, kScratchDoubleReg);
      if (instr->InputCount() == 2) {
        ASSEMBLE_SIMD_IMM_INSTR(pblendw, kScratchDoubleReg, 1, 0x55);
        src2 = kScratchDoubleReg;
      }
      __ pblendw(dst, kScratchDoubleReg, 0xaa);
      __ packusdw(dst, src2);
      break;
    }
    case kX64S8x16UnzipHigh: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src2 = dst;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (instr->InputCount() == 2) {
        ASSEMBLE_SIMD_INSTR(movups, kScratchDoubleReg, 1);
        __ psrlw(kScratchDoubleReg, 8);
        src2 = kScratchDoubleReg;
      }
      __ psrlw(dst, 8);
      __ packuswb(dst, src2);
      break;
    }
    case kX64S8x16UnzipLow: {
      XMMRegister dst = i.OutputSimd128Register();
      XMMRegister src2 = dst;
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      if (instr->InputCount() == 2) {
        ASSEMBLE_SIMD_INSTR(movups, kScratchDoubleReg, 1);
        __ psllw(kScratchDoubleReg, 8);
        __ psrlw(kScratchDoubleReg, 8);
        src2 = kScratchDoubleReg;
      }
      __ psllw(dst, 8);
      __ psrlw(dst, 8);
      __ packuswb(dst, src2);
      break;
    }
    case kX64S8x16TransposeLow: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ psllw(dst, 8);
      if (instr->InputCount() == 1) {
        __ movups(kScratchDoubleReg, dst);
      } else {
        DCHECK_EQ(2, instr->InputCount());
        ASSEMBLE_SIMD_INSTR(movups, kScratchDoubleReg, 1);
        __ psllw(kScratchDoubleReg, 8);
      }
      __ psrlw(dst, 8);
      __ por(dst, kScratchDoubleReg);
      break;
    }
    case kX64S8x16TransposeHigh: {
      XMMRegister dst = i.OutputSimd128Register();
      DCHECK_EQ(dst, i.InputSimd128Register(0));
      __ psrlw(dst, 8);
      if (instr->InputCount() == 1) {
        __ movups(kScratchDoubleReg, dst);
      } else {
        DCHECK_EQ(2, instr->InputCount());
        ASSEMBLE_SIMD_INSTR(movups, kScratchDoubleReg, 1);
        __ psrlw(kScratchDoubleReg, 8);
      }
      __ psllw(kScratchDoubleReg, 8);
      __ por(dst, kScratchDoubleReg);
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
        int8_t shuffle_mask = arch_opcode == kX64S8x4Reverse ? 0xB1 : 0x1B;
        __ pshuflw(dst, dst, shuffle_mask);
        __ pshufhw(dst, dst, shuffle_mask);
      }
      __ movaps(kScratchDoubleReg, dst);
      __ psrlw(kScratchDoubleReg, 8);
      __ psllw(dst, 8);
      __ por(dst, kScratchDoubleReg);
      break;
    }
    case kX64S1x2AnyTrue:
    case kX64S1x4AnyTrue:
    case kX64S1x8AnyTrue:
    case kX64S1x16AnyTrue: {
      CpuFeatureScope sse_scope(tasm(), SSE4_1);
      Register dst = i.OutputRegister();
      XMMRegister src = i.InputSimd128Register(0);
      Register tmp = i.TempRegister(0);
      __ xorq(tmp, tmp);
      __ movq(dst, Immediate(1));
      __ ptest(src, src);
      __ cmovq(zero, dst, tmp);
      break;
    }
    // Need to split up all the different lane structures because the
    // comparison instruction used matters, e.g. given 0xff00, pcmpeqb returns
    // 0x0011, pcmpeqw returns 0x0000, ptest will set ZF to 0 and 1
    // respectively.
    case kX64S1x2AllTrue: {
      ASSEMBLE_SIMD_ALL_TRUE(pcmpeqq);
      break;
    }
    case kX64S1x4AllTrue: {
      ASSEMBLE_SIMD_ALL_TRUE(Pcmpeqd);
      break;
    }
    case kX64S1x8AllTrue: {
      ASSEMBLE_SIMD_ALL_TRUE(Pcmpeqw);
      break;
    }
    case kX64S1x16AllTrue: {
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
