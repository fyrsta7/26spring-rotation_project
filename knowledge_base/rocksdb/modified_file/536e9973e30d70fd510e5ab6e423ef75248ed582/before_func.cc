void VectorRep::Insert(KeyHandle handle) {
  auto* key = static_cast<char*>(handle);
  assert(!Contains(key));
  WriteLock l(&rwlock_);
  assert(!immutable_);
  bucket_->push_back(key);
}
