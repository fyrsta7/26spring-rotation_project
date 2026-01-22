absl::Status GlContext::CreateContextInternal(
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE external_context, int webgl_version) {
  ABSL_CHECK(webgl_version == 1 || webgl_version == 2);

  EmscriptenWebGLContextAttributes attrs;
  emscripten_webgl_init_context_attributes(&attrs);
  attrs.explicitSwapControl = 0;
  attrs.depth = 1;
  attrs.stencil = 0;
  attrs.antialias = 0;
  attrs.majorVersion = webgl_version;
  attrs.minorVersion = 0;

  // This flag tells the page compositor that the image written to the canvas
  // uses premultiplied alpha, and so can be used directly for compositing.
  // Without this, it needs to make an additional full-canvas rendering pass.
  attrs.premultipliedAlpha = 1;

  // TODO: Investigate this option in more detail, esp. on Safari.
  attrs.preserveDrawingBuffer = 0;

  // Quick patch for -s DISABLE_DEPRECATED_FIND_EVENT_TARGET_BEHAVIOR so it also
  // looks for our #canvas target in Module.canvas, where we expect it to be.
  // -s OFFSCREENCANVAS_SUPPORT=1 will no longer work with this under the new
  // event target behavior, but it was never supposed to be tapping into our
  // canvas anyways. See b/278155946 for more background.
  EM_ASM({ specialHTMLTargets["#canvas"] = Module.canvas; });
  EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_handle =
      emscripten_webgl_create_context("#canvas", &attrs);

  // Check for failure
  if (context_handle <= 0) {
    ABSL_LOG(INFO) << "Couldn't create webGL " << webgl_version << " context.";
    return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "emscripten_webgl_create_context() returned error "
           << context_handle;
  } else {
    emscripten_webgl_get_context_attributes(context_handle, &attrs);
    webgl_version = attrs.majorVersion;
  }
  context_ = context_handle;
  attrs_ = attrs;
  // We can't always rely on GL_MAJOR_VERSION and GL_MINOR_VERSION, since
  // GLES 2 does not have them, so let's set the major version here at least.
  // WebGL 1.0 maps to GLES 2.0 and WebGL 2.0 maps to GLES 3.0, so we add 1.
  gl_major_version_ = webgl_version + 1;
  return absl::OkStatus();
}
