void OptimizingCompilerThread::FlushOutputQueue(bool restore_function_code) {
  base::LockGuard<base::Mutex> access_output_queue_(&output_queue_mutex_);
  OptimizedCompileJob* job;
  while (output_queue_.Dequeue(&job)) {
    // OSR jobs are dealt with separately.
    if (!job->info()->is_osr()) {
      DisposeOptimizedCompileJob(job, restore_function_code);
    }
  }
}
