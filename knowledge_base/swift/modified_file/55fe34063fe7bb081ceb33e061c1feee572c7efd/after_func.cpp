raw_ostream &operator<<(raw_ostream &OS, SILConstant t) {
  t.print(OS);
  return OS;
}
