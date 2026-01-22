static llvm::Constant *getRuntimeFn(IRGenModule &IGM,
                      llvm::Constant *&cache,
                      char const *name,
                      llvm::CallingConv::ID cc,
                      std::initializer_list<llvm::Type*> retTypes,
                      std::initializer_list<llvm::Type*> argTypes,
                      std::initializer_list<Attribute::AttrKind> attrs
                         = std::initializer_list<Attribute::AttrKind>()) {
  if (cache)
    return cache;
  
  llvm::Type *retTy;
  if (retTypes.size() == 1)
    retTy = *retTypes.begin();
  else
    retTy = llvm::StructType::get(IGM.LLVMContext,
                                  {retTypes.begin(), retTypes.end()},
                                  /*packed*/ false);
  auto fnTy = llvm::FunctionType::get(retTy,
                                      {argTypes.begin(), argTypes.end()},
                                      /*isVararg*/ false);

  cache = IGM.Module.getOrInsertFunction(name, fnTy);

  // Add any function attributes and set the calling convention.
  if (auto fn = dyn_cast<llvm::Function>(cache)) {
    fn->setCallingConv(cc);

    for (auto Attr : attrs)
      fn->addFnAttr(Attr);
  }

  return cache;
}
