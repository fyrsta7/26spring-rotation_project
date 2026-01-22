  bool optimize(Expr &expr) override {
    if (expr->type == NodeType::load || expr->type == NodeType::store) {
      auto &ptr = expr._pointer();
      auto &addr_node = expr._pointer()._address();
      bool all_same = true;

      for (int i = 0; i < addr_node->lanes; i++) {
        if (addr_node->snode_ptr(i) != addr_node->snode_ptr(0))
          all_same = false;
      }

      bool incremental = true;
      bool regular_elements = true;
      int offset_start;
      int offset_inc;

      // only consider the last one.
      auto &index_node = expr._pointer()->ch.back();

      if (index_node->type != NodeType::index) {
        return false;
      }
      bool indirect = false;
      if (kernel->program.current_snode->type == SNodeType::indirect &&
          index_node->index_id(0) == kernel->program.current_snode->index_id) {
        TC_INFO("Skipped optimization since it is indirect.");
        indirect = true;
      }

      // TODO: check non-last indices are uniform
      if (index_node->type == NodeType::index) {
        offset_start = index_node->index_offset(0);
        offset_inc = index_node->index_offset(1) - offset_start;
        for (int i = 0; i + 1 < addr_node->lanes; i++) {
          if (index_node->index_offset(i) + offset_inc !=
              index_node->index_offset(i + 1)) {
            incremental = false;
          }
        }
      } else {
        // computed indices
        // case I: adapter load
        incremental = true;
        offset_start = 0;  // useless
        offset_inc = 0;

        // second DFS to match two lanes
        if (all_same) {
          for (int i = 0; i + 1 < index_node->lanes; i++) {
            auto ret = AddressAnalyzer(i, i + 1).run(index_node);
            if (!ret.first) {
              incremental = false;
            } else {
              if (i == 0) {
                offset_inc = ret.second;
              } else if (offset_inc != ret.second) {
                incremental = false;
                break;
              }
            }
          }
        }
      }

      for (int i = 0; i < addr_node->lanes; i++) {
        auto p = addr_node->snode_ptr(i)->parent;
        if (p != addr_node->snode_ptr(0)->parent)
          regular_elements = false;
        if (p->child_id(addr_node->snode_ptr(i)) != i)
          regular_elements = false;
      }

      auto snode = addr_node->snode_ptr(0);
      // continuous index, same element
      bool vpointer_case_1 = incremental && offset_start == 0 &&
                             offset_inc == 1 &&
                             (snode->parent->type == SNodeType::fixed ||
                              snode->parent->type == SNodeType::dynamic) &&
                             all_same;
      // continuous element, same index
      bool vpointer_case_2 = regular_elements && incremental && offset_inc == 0;
      bool vpointer = !indirect && (vpointer_case_1 || vpointer_case_2);

      if (vpointer) {
        // replace load with vload
        if (expr->type == NodeType::load) {
          TC_INFO("Optimized load to vloadu");
          auto vload = Expr::create(NodeType::vload, addr_node);
          vload->ch.resize(ptr->ch.size());
          for (int i = 1; i < (int)ptr->ch.size(); i++) {
            auto c = Expr::copy_from(ptr->ch[i]);
            TC_ASSERT(c->lanes == kernel->program.config.simd_width);
            if (c->type == NodeType::index)
              c->set_lanes(1);
            vload->ch[i] = c;
          }
          vload->set_similar(expr);
          expr.set(vload);
          return true;
        } else {
          TC_INFO("Optimized store to vstoreu");
          auto vstore = Expr::create(NodeType::vstore, addr_node, expr->ch[1]);
          vstore->ch.resize(ptr->ch.size() + 1);
          for (int i = 1; i < (int)ptr->ch.size(); i++) {
            auto c = Expr::copy_from(ptr->ch[i]);
            TC_ASSERT(c->lanes == kernel->program.config.simd_width);
            if (c->type == NodeType::index)
              c->set_lanes(1);
            vstore->ch[i + 1] = c;
          }
          vstore->set_similar(expr);
          expr.set(vstore);
          return true;
        }
      }
    }
    return false;
  }
