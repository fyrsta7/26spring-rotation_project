    Py_INCREF(python_completion);
  }

  ~MonCommandCompletion() override
  {
    if (python_completion) {
      // Usually do this in finish(): this path is only for if we're
      // being destroyed without completing.
      Gil gil(pThreadState, true);
      Py_DECREF(python_completion);
      python_completion = nullptr;
    }
  }

  void finish(int r) override
  {
    ceph_assert(python_completion != nullptr);

    dout(10) << "MonCommandCompletion::finish()" << dendl;
    {
      // Scoped so the Gil is released before calling notify_all()
      // Create new thread state because this is called via the MonClient
      // Finisher, not the PyModules finisher.
      Gil gil(pThreadState, true);

      auto set_fn = PyObject_GetAttrString(python_completion, "complete");
      ceph_assert(set_fn != nullptr);

      auto pyR = PyLong_FromLong(r);
      auto pyOutBl = PyUnicode_FromString(outbl.to_str().c_str());
      auto pyOutS = PyUnicode_FromString(outs.c_str());
      auto args = PyTuple_Pack(3, pyR, pyOutBl, pyOutS);
      Py_DECREF(pyR);
      Py_DECREF(pyOutBl);
