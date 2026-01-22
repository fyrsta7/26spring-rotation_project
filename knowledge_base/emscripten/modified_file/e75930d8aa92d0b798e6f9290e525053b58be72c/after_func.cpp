void traversePre(Ref node, std::function<void (Ref)> visit) {
  visit(node);
  std::vector<TraverseInfo> stack;
  stack.push_back({ node, 0 });
  while (1) {
    TraverseInfo& top = stack.back();
    if (top.index < top.node->size()) {
      Ref sub = top.node[top.index];
      top.index++;
      if (visitable(sub)) {
        visit(sub);
        stack.push_back({ sub, 0 });
      }
    } else {
      if (stack.size() == 1) break;
      stack.pop_back();
    }
  }
}
