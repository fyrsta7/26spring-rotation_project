void Assembler::CheckLabelLinkChain(Label const * label) {
#ifdef DEBUG
  if (label->is_linked()) {
    static const int kMaxLinksToCheck = 256;  // Avoid O(n2) behaviour.
    int links_checked = 0;
    int linkoffset = label->pos();
    bool end_of_chain = false;
    while (!end_of_chain) {
      if (++links_checked > kMaxLinksToCheck) break;
      Instruction * link = InstructionAt(linkoffset);
      int linkpcoffset = link->ImmPCOffset();
      int prevlinkoffset = linkoffset + linkpcoffset;

      end_of_chain = (linkoffset == prevlinkoffset);
      linkoffset = linkoffset + linkpcoffset;
    }
  }
#endif
}
