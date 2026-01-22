// text mangling will result.
bool BTypeVisitor::TraverseCallExpr(CallExpr *Call) {
  for (auto child : Call->children())
    if (!TraverseStmt(child))
      return false;
  if (!WalkUpFromCallExpr(Call))
    return false;
  return true;
}

// convert calls of the type:
//  table.foo(&key)
// to:
//  bpf_table_foo_elem(bpf_pseudo_fd(table), &key [,&leaf])
bool BTypeVisitor::VisitCallExpr(CallExpr *Call) {
  // make sure node is a reference to a bpf table, which is assured by the
  // presence of the section("maps/<typename>") GNU __attribute__
  if (MemberExpr *Memb = dyn_cast<MemberExpr>(Call->getCallee()->IgnoreImplicit())) {
    StringRef memb_name = Memb->getMemberDecl()->getName();
    if (DeclRefExpr *Ref = dyn_cast<DeclRefExpr>(Memb->getBase())) {
      if (SectionAttr *A = Ref->getDecl()->getAttr<SectionAttr>()) {
        if (!A->getName().startswith("maps"))
          return true;

        string args = rewriter_.getRewrittenText(expansionRange(SourceRange(Call->getArg(0)->getLocStart(),
                                                   Call->getArg(Call->getNumArgs() - 1)->getLocEnd())));

        // find the table fd, which was opened at declaration time
        TableStorage::iterator desc;
        Path local_path({fe_.id(), Ref->getDecl()->getName()});
        Path global_path({Ref->getDecl()->getName()});
        if (!fe_.table_storage().Find(local_path, desc)) {
          if (!fe_.table_storage().Find(global_path, desc)) {
            error(Ref->getLocEnd(), "bpf_table %0 failed to open") << Ref->getDecl()->getName();
            return false;
          }
        }
        string fd = to_string(desc->second.fd);
        string prefix, suffix;
        string txt;
        auto rewrite_start = Call->getLocStart();
        auto rewrite_end = Call->getLocEnd();
        if (memb_name == "lookup_or_init") {
          string name = Ref->getDecl()->getName();
          string arg0 = rewriter_.getRewrittenText(expansionRange(Call->getArg(0)->getSourceRange()));
          string arg1 = rewriter_.getRewrittenText(expansionRange(Call->getArg(1)->getSourceRange()));
          string lookup = "bpf_map_lookup_elem_(bpf_pseudo_fd(1, " + fd + ")";
          string update = "bpf_map_update_elem_(bpf_pseudo_fd(1, " + fd + ")";
          txt  = "({typeof(" + name + ".leaf) *leaf = " + lookup + ", " + arg0 + "); ";
          txt += "if (!leaf) {";
          txt += " " + update + ", " + arg0 + ", " + arg1 + ", BPF_NOEXIST);";
          txt += " leaf = " + lookup + ", " + arg0 + ");";
          txt += " if (!leaf) return 0;";
          txt += "}";
          txt += "leaf;})";
        } else if (memb_name == "increment") {
          string name = Ref->getDecl()->getName();
          string arg0 = rewriter_.getRewrittenText(expansionRange(Call->getArg(0)->getSourceRange()));
          string lookup = "bpf_map_lookup_elem_(bpf_pseudo_fd(1, " + fd + ")";
          string update = "bpf_map_update_elem_(bpf_pseudo_fd(1, " + fd + ")";
          txt  = "({ typeof(" + name + ".key) _key = " + arg0 + "; ";
          txt += "typeof(" + name + ".leaf) *_leaf = " + lookup + ", &_key); ";
          txt += "if (_leaf) (*_leaf)++; ";
          if (desc->second.type == BPF_MAP_TYPE_HASH) {
            txt += "else { typeof(" + name + ".leaf) _zleaf; memset(&_zleaf, 0, sizeof(_zleaf)); ";
            txt += "_zleaf++; ";
            txt += update + ", &_key, &_zleaf, BPF_NOEXIST); } ";
          }
          txt += "})";
        } else if (memb_name == "perf_submit") {
          string name = Ref->getDecl()->getName();
          string arg0 = rewriter_.getRewrittenText(expansionRange(Call->getArg(0)->getSourceRange()));
          string args_other = rewriter_.getRewrittenText(expansionRange(SourceRange(Call->getArg(1)->getLocStart(),
                                                           Call->getArg(2)->getLocEnd())));
          txt = "bpf_perf_event_output(" + arg0 + ", bpf_pseudo_fd(1, " + fd + ")";
          txt += ", CUR_CPU_IDENTIFIER, " + args_other + ")";
        } else if (memb_name == "perf_submit_skb") {
          string skb = rewriter_.getRewrittenText(expansionRange(Call->getArg(0)->getSourceRange()));
          string skb_len = rewriter_.getRewrittenText(expansionRange(Call->getArg(1)->getSourceRange()));
          string meta = rewriter_.getRewrittenText(expansionRange(Call->getArg(2)->getSourceRange()));
          string meta_len = rewriter_.getRewrittenText(expansionRange(Call->getArg(3)->getSourceRange()));
          txt = "bpf_perf_event_output(" +
            skb + ", " +
            "bpf_pseudo_fd(1, " + fd + "), " +
            "((__u64)" + skb_len + " << 32) | BPF_F_CURRENT_CPU, " +
            meta + ", " +
            meta_len + ");";
        } else if (memb_name == "get_stackid") {
          if (desc->second.type == BPF_MAP_TYPE_STACK_TRACE) {
            string arg0 =
                rewriter_.getRewrittenText(expansionRange(Call->getArg(0)->getSourceRange()));
            txt = "bpf_get_stackid(";
            txt += "bpf_pseudo_fd(1, " + fd + "), " + arg0;
            rewrite_end = Call->getArg(0)->getLocEnd();
            } else {
              error(Call->getLocStart(), "get_stackid only available on stacktrace maps");
              return false;
            }
        } else {
          if (memb_name == "lookup") {
            prefix = "bpf_map_lookup_elem";
            suffix = ")";
          } else if (memb_name == "update") {
            prefix = "bpf_map_update_elem";
            suffix = ", BPF_ANY)";
          } else if (memb_name == "insert") {
            if (desc->second.type == BPF_MAP_TYPE_ARRAY) {
              warning(Call->getLocStart(), "all element of an array already exist; insert() will have no effect");
            }
            prefix = "bpf_map_update_elem";
            suffix = ", BPF_NOEXIST)";
          } else if (memb_name == "delete") {
            prefix = "bpf_map_delete_elem";
            suffix = ")";
          } else if (memb_name == "call") {
            prefix = "bpf_tail_call_";
            suffix = ")";
          } else if (memb_name == "perf_read") {
            prefix = "bpf_perf_event_read";
            suffix = ")";
          } else {
            error(Call->getLocStart(), "invalid bpf_table operation %0") << memb_name;
            return false;
          }
          prefix += "((void *)bpf_pseudo_fd(1, " + fd + "), ";

          txt = prefix + args + suffix;
        }
        if (!rewriter_.isRewritable(rewrite_start) || !rewriter_.isRewritable(rewrite_end)) {
          error(Call->getLocStart(), "cannot use map function inside a macro");
          return false;
        }
        rewriter_.ReplaceText(expansionRange(SourceRange(rewrite_start, rewrite_end)), txt);
        return true;
      }
    }
  } else if (Call->getCalleeDecl()) {
    NamedDecl *Decl = dyn_cast<NamedDecl>(Call->getCalleeDecl());
    if (!Decl) return true;
    if (AsmLabelAttr *A = Decl->getAttr<AsmLabelAttr>()) {
      // Functions with the tag asm("llvm.bpf.extra") are implemented in the
      // rewriter rather than as a macro since they may also include nested
      // rewrites, and clang::Rewriter does not support rewrites in macros,
      // unless one preprocesses the entire source file.
      if (A->getLabel() == "llvm.bpf.extra") {
        if (!rewriter_.isRewritable(Call->getLocStart())) {
          error(Call->getLocStart(), "cannot use builtin inside a macro");
          return false;
        }

        vector<string> args;
        for (auto arg : Call->arguments())
          args.push_back(rewriter_.getRewrittenText(expansionRange(arg->getSourceRange())));

        string text;
        if (Decl->getName() == "incr_cksum_l3") {
          text = "bpf_l3_csum_replace_(" + fn_args_[0]->getName().str() + ", (u64)";
          text += args[0] + ", " + args[1] + ", " + args[2] + ", sizeof(" + args[2] + "))";
          rewriter_.ReplaceText(expansionRange(Call->getSourceRange()), text);
        } else if (Decl->getName() == "incr_cksum_l4") {
          text = "bpf_l4_csum_replace_(" + fn_args_[0]->getName().str() + ", (u64)";
          text += args[0] + ", " + args[1] + ", " + args[2];
          text += ", ((" + args[3] + " & 0x1) << 4) | sizeof(" + args[2] + "))";
          rewriter_.ReplaceText(expansionRange(Call->getSourceRange()), text);
        } else if (Decl->getName() == "bpf_trace_printk") {
          checkFormatSpecifiers(args[0], Call->getArg(0)->getLocStart());
          //  #define bpf_trace_printk(fmt, args...)
          //    ({ char _fmt[] = fmt; bpf_trace_printk_(_fmt, sizeof(_fmt), args...); })
          text = "({ char _fmt[] = " + args[0] + "; bpf_trace_printk_(_fmt, sizeof(_fmt)";
          if (args.size() <= 1) {
            text += "); })";
            rewriter_.ReplaceText(expansionRange(Call->getSourceRange()), text);
          } else {
            rewriter_.ReplaceText(expansionRange(SourceRange(Call->getLocStart(), Call->getArg(0)->getLocEnd())), text);
            rewriter_.InsertTextAfter(Call->getLocEnd(), "); }");
          }
        } else if (Decl->getName() == "bpf_num_cpus") {
          int numcpu = sysconf(_SC_NPROCESSORS_ONLN);
          if (numcpu <= 0)
            numcpu = 1;
          text = to_string(numcpu);
          rewriter_.ReplaceText(expansionRange(Call->getSourceRange()), text);
        } else if (Decl->getName() == "bpf_usdt_readarg_p") {
          text = "({ u64 __addr = 0x0; ";
          text += "_bpf_readarg_" + current_fn_ + "_" + args[0] + "(" +
                  args[1] + ", &__addr, sizeof(__addr));";
          text += "bpf_probe_read(" + args[2] + ", " + args[3] +
                  ", (void *)__addr);";
          text += "})";
          rewriter_.ReplaceText(expansionRange(Call->getSourceRange()), text);
        } else if (Decl->getName() == "bpf_usdt_readarg") {
          text = "_bpf_readarg_" + current_fn_ + "_" + args[0] + "(" + args[1] +
                 ", " + args[2] + ", sizeof(*(" + args[2] + ")))";
          rewriter_.ReplaceText(expansionRange(Call->getSourceRange()), text);
        }
