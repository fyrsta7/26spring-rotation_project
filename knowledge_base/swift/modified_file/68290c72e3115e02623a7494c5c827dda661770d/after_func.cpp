void CompilerInstance::performParse() {
  const SourceFileKind Kind = Invocation.getInputKind();
  Identifier ID = Context->getIdentifier(Invocation.getModuleName());
  MainModule = new (*Context) Module(ID, *Context);
  Context->LoadedModules[ID.str()] = MainModule;

  if (Kind == SourceFileKind::SIL) {
    assert(BufferIDs.size() == 1);
    assert(MainBufferID != NO_SUCH_BUFFER);
    createSILModule();
  }

  if (Kind == SourceFileKind::REPL) {
    auto *SingleInputFile =
      new (*Context) SourceFile(*MainModule, Kind, {},
                                Invocation.getParseStdlib());
    MainModule->addFile(*SingleInputFile);
    return;
  }

  std::unique_ptr<DelayedParsingCallbacks> DelayedCB;
  if (Invocation.isCodeCompletion()) {
    DelayedCB.reset(
        new CodeCompleteDelayedCallbacks(SourceMgr.getCodeCompletionLoc()));
  } else if (Invocation.isDelayedFunctionBodyParsing()) {
    DelayedCB.reset(new AlwaysDelayedCallbacks);
  }

  PersistentParserState PersistentState;

  // Make sure the main file is the first file in the module. This may only be
  // a source file, or it may be a SIL file, which requires pumping the parser.
  // We parse it last, though, to make sure that it can use decls from other
  // files in the module.
  if (MainBufferID != NO_SUCH_BUFFER) {
    assert(Kind == SourceFileKind::Main || Kind == SourceFileKind::SIL);

    if (Kind == SourceFileKind::Main)
      SourceMgr.setHashbangBufferID(MainBufferID);

    auto *SingleInputFile =
      new (*Context) SourceFile(*MainModule, Kind, MainBufferID,
                                Invocation.getParseStdlib());
    MainModule->addFile(*SingleInputFile);

    if (MainBufferID == PrimaryBufferID)
      PrimarySourceFile = SingleInputFile;
  }

  bool hadLoadError = false;

  // Parse all the library files first.
  for (size_t i = 0, e = BufferIDs.size(); i < e; ++i) {
    auto BufferID = BufferIDs[i];
    if (BufferID == MainBufferID)
      continue;

    auto Buffer = SourceMgr.getLLVMSourceMgr().getMemoryBuffer(BufferID);
    if (SerializedModuleLoader::isSerializedAST(Buffer->getBuffer())) {
      std::unique_ptr<llvm::MemoryBuffer> Input(
        llvm::MemoryBuffer::getMemBuffer(Buffer->getBuffer(),
                                         Buffer->getBufferIdentifier(),
                                         false));
      if (!SML->loadAST(*MainModule, SourceLoc(), std::move(Input)))
        hadLoadError = true;
      continue;
    }

    auto *NextInput = new (*Context) SourceFile(*MainModule,
                                                SourceFileKind::Library,
                                                BufferID,
                                                Invocation.getParseStdlib());
    MainModule->addFile(*NextInput);

    if (BufferID == PrimaryBufferID)
      PrimarySourceFile = NextInput;

    bool Done;
    parseIntoSourceFile(*NextInput, BufferID, &Done, nullptr,
                        &PersistentState, DelayedCB.get());
    assert(Done && "Parser returned early?");
    (void) Done;

    performNameBinding(*NextInput);
  }

  if (hadLoadError)
    return;

  // Parse the main file last.
  if (MainBufferID != NO_SUCH_BUFFER) {
    SourceFile &MainFile = MainModule->getMainSourceFile(Kind);
    SILParserState SILContext(TheSILModule.get());

    unsigned CurTUElem = 0;
    bool Done;
    do {
      // Pump the parser multiple times if necessary.  It will return early
      // after parsing any top level code in a main module, or in SIL mode when
      // there are chunks of swift decls (e.g. imports and types) interspersed
      // with 'sil' definitions.
      parseIntoSourceFile(MainFile, MainFile.getBufferID().getValue(), &Done,
                          TheSILModule ? &SILContext : nullptr,
                          &PersistentState, DelayedCB.get());
      if (!Invocation.getParseOnly() && (PrimaryBufferID == NO_SUCH_BUFFER ||
                                         MainBufferID == PrimaryBufferID))
        performTypeChecking(MainFile, PersistentState.getTopLevelContext(),
                            CurTUElem);
      CurTUElem = MainFile.Decls.size();
    } while (!Done);
  }

  if (!Invocation.getParseOnly()) {
    // Type-check each top-level input besides the main source file.
    auto InputSourceFiles = MainModule->getFiles().slice(0, BufferIDs.size());
    for (auto File : InputSourceFiles)
      if (auto SF = dyn_cast<SourceFile>(File))
        if (PrimaryBufferID == NO_SUCH_BUFFER ||
            (SF->getBufferID().hasValue() &&
             SF->getBufferID().getValue() == PrimaryBufferID))
            performTypeChecking(*SF, PersistentState.getTopLevelContext());

    // If there were no source files, we should still record known protocols.
    if (Context->getStdlibModule())
      Context->recordKnownProtocols(Context->getStdlibModule());
  }

  if (DelayedCB) {
    performDelayedParsing(MainModule, PersistentState,
                          Invocation.getCodeCompletionFactory());
  }
}
