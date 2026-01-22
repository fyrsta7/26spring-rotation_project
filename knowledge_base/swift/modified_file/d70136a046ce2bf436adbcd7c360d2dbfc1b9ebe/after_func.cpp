void IRGenModule::emitProtocolDecl(ProtocolDecl *protocol) {
  // If the protocol is Objective-C-compatible, go through the path that
  // produces an ObjC-compatible protocol_t.
  if (protocol->isObjC()) {
    // Native ObjC protocols are emitted on-demand in ObjC and uniqued by the
    // runtime; we don't need to try to emit a unique descriptor symbol for them.
    if (protocol->hasClangNode())
      return;
    
    getObjCProtocolGlobalVars(protocol);
    return;
  }
  
  ProtocolDescriptorBuilder builder(*this, protocol);
  builder.layout();
  auto init = builder.getInit();

  auto var = cast<llvm::GlobalVariable>(
                       getAddrOfProtocolDescriptor(protocol, ForDefinition));
  var->setConstant(true);
  var->setInitializer(init);
}
