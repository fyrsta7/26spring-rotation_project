    MDSCacheObject::dump(f);
  }
  if (flags & DUMP_ITEMS) {
    f->open_array_section("dentries");
    for (auto &p : items) {
      CDentry *dn = p.second;
      f->open_object_section("dentry");
      dn->dump(f);
      f->close_section();
    }
    f->close_section();
  }
}

void CDir::dump_load(Formatter *f)
{
  f->dump_stream("path") << get_path();
  f->dump_stream("dirfrag") << dirfrag();

  f->open_object_section("pop_me");
  pop_me.dump(f);
  f->close_section();

  f->open_object_section("pop_nested");
  pop_nested.dump(f);
  f->close_section();

  f->open_object_section("pop_auth_subtree");
  pop_auth_subtree.dump(f);
  f->close_section();

  f->open_object_section("pop_auth_subtree_nested");
  pop_auth_subtree_nested.dump(f);
  f->close_section();
