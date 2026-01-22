          ->getMethod<alias_ref<JMountItem>(jint)>("deleteMountItem");

  return deleteInstruction(javaUIManager, mutation.oldChildShadowView.tag);
}

local_ref<JMountItem::javaobject> createRemoveAndDeleteMultiMountItem(
  const jni::global_ref<jobject> &javaUIManager,
  const std::vector<RemoveDeleteMetadata> &metadata) {

  auto env = Environment::current();
  auto removeAndDeleteArray = env->NewIntArray(metadata.size()*4);
  int position = 0;
  jint temp[4];
  for (const auto& x : metadata) {
    temp[0] = x.tag;
    temp[1] = x.parentTag;
    temp[2] = x.index;
    temp[3] = (x.shouldRemove ? 1 : 0) | (x.shouldDelete ? 2 : 0);
    env->SetIntArrayRegion(removeAndDeleteArray, position, 4, temp);
    position += 4;
  }

  static auto removeDeleteMultiInstruction =
    jni::findClassStatic(UIManagerJavaDescriptor)
      ->getMethod<alias_ref<JMountItem>(jintArray)>("removeDeleteMultiMountItem");

  auto ret = removeDeleteMultiInstruction(javaUIManager, removeAndDeleteArray);

  // It is not strictly necessary to manually delete the ref here, in this particular case.
  // If JNI memory is being allocated in a loop, it's easy to overload the localref table
  // and crash; this is not possible in this case since the JNI would automatically clear this
  // ref when it goes out of scope, anyway. However, this is being left here as a reminder of
