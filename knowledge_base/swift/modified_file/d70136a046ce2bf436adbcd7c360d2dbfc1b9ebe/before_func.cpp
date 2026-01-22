void IRGenModule::emitProtocolDecl(ProtocolDecl *protocol) {
  // If the protocol is Objective-C-compatible, go through the path that
  // produces an ObjC-compatible protocol_t.
  if (protocol->isObjC()) {
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
