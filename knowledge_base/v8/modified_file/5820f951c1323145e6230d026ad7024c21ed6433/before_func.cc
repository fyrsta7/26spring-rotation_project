void FastCodeGenerator::Generate(FunctionLiteral* fun) {
  function_ = fun;
  SetFunctionPosition(fun);

  __ push(ebp);  // Caller's frame pointer.
  __ mov(ebp, esp);
  __ push(esi);  // Callee's context.
  __ push(edi);  // Callee's JS Function.

  { Comment cmnt(masm_, "[ Allocate locals");
    int locals_count = fun->scope()->num_stack_slots();
    for (int i = 0; i < locals_count; i++) {
      __ push(Immediate(Factory::undefined_value()));
    }
  }

  bool function_in_register = true;

  Variable* arguments = fun->scope()->arguments()->AsVariable();
  if (arguments != NULL) {
    // Function uses arguments object.
    Comment cmnt(masm_, "[ Allocate arguments object");
    __ push(edi);
    // Receiver is just before the parameters on the caller's stack.
    __ lea(edx, Operand(ebp, StandardFrameConstants::kCallerSPOffset +
                                 fun->num_parameters() * kPointerSize));
    __ push(edx);
    __ push(Immediate(Smi::FromInt(fun->num_parameters())));
    // Arguments to ArgumentsAccessStub:
    //   function, receiver address, parameter count.
    // The stub will rewrite receiever and parameter count if the previous
    // stack frame was an arguments adapter frame.
    ArgumentsAccessStub stub(ArgumentsAccessStub::NEW_OBJECT);
    __ CallStub(&stub);
    __ mov(Operand(ebp, SlotOffset(arguments->slot())), eax);
    Slot* dot_arguments_slot =
        fun->scope()->arguments_shadow()->AsVariable()->slot();
    __ mov(Operand(ebp, SlotOffset(dot_arguments_slot)), eax);

    function_in_register = false;
  }

  // Possibly allocate a local context.
  if (fun->scope()->num_heap_slots() > 0) {
    Comment cmnt(masm_, "[ Allocate local context");
    if (function_in_register) {
      // Argument to NewContext is the function, still in edi.
      __ push(edi);
    } else {
      // Argument to NewContext is the function, no longer in edi.
      __ push(Operand(ebp, JavaScriptFrameConstants::kFunctionOffset));
    }
    __ CallRuntime(Runtime::kNewContext, 1);
    // Context is returned in both eax and esi.  It replaces the context
    // passed to us.  It's saved in the stack and kept live in esi.
    __ mov(Operand(ebp, StandardFrameConstants::kContextOffset), esi);
#ifdef DEBUG
    // Assert we do not have to copy any parameters into the context.
    for (int i = 0, len = fun->scope()->num_parameters(); i < len; i++) {
      Slot* slot = fun->scope()->parameter(i)->slot();
      ASSERT(slot != NULL && slot->type() != Slot::CONTEXT);
    }
#endif
  }

  { Comment cmnt(masm_, "[ Declarations");
    VisitDeclarations(fun->scope()->declarations());
  }

  { Comment cmnt(masm_, "[ Stack check");
    Label ok;
    ExternalReference stack_limit =
        ExternalReference::address_of_stack_limit();
    __ cmp(esp, Operand::StaticVariable(stack_limit));
    __ j(above_equal, &ok, taken);
    StackCheckStub stub;
    __ CallStub(&stub);
    __ bind(&ok);
  }

  if (FLAG_trace) {
    __ CallRuntime(Runtime::kTraceEnter, 0);
  }

  { Comment cmnt(masm_, "[ Body");
    ASSERT(loop_depth() == 0);
    VisitStatements(fun->body());
    ASSERT(loop_depth() == 0);
  }

  { Comment cmnt(masm_, "[ return <undefined>;");
    // Emit a 'return undefined' in case control fell off the end of the body.
    __ mov(eax, Factory::undefined_value());
    EmitReturnSequence(function_->end_position());
  }
}
