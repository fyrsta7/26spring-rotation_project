static void executeApplicationScript(
    const RefPtr<Bridge>& bridge,
    const std::string script,
    const std::string sourceUri) {
  try {
    // Execute the application script and collect/dispatch any native calls that might have occured
    bridge->executeApplicationScript(script, sourceUri);
    bridge->flush();
  } catch (...) {
    translatePendingCppExceptionToJavaException();
  }
}
