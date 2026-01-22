
std::string CHJIT::getMangledName(const std::string & name_to_mangle) const
{
    std::string mangled_name;
    llvm::raw_string_ostream mangled_name_stream(mangled_name);
    llvm::Mangler::getNameWithPrefix(mangled_name_stream, name_to_mangle, layout);
    mangled_name_stream.flush();

    return mangled_name;
}

void CHJIT::runOptimizationPassesOnModule(llvm::Module & module) const
{
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

