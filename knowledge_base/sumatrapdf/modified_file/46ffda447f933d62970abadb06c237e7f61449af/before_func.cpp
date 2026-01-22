
TreeItem* TocItem::ChildAt(int n) {
    if (n == 0) {
        currChild = child;
        currChildNo = 0;
        return child;
    }
    // speed up sequential iteration over children
    if (currChild != nullptr && n == currChildNo + 1) {
        auto res = currChild;
        currChild = currChild->next;
        ++currChildNo;
        return res;
    }
    auto node = child;
    while (n > 0) {
        n--;
        node = node->next;
    }
    return node;
