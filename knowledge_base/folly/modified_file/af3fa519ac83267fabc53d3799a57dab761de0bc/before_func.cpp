}

void EventBase::loopForever() {
  // Update the notification queue event to treat it as a normal (non-internal)
  // event.  The notification queue event always remains installed, and the main
  // loop won't exit with it installed.
  fnRunner_->stopConsuming();
  fnRunner_->startConsuming(this, queue_.get());

  bool ret = loop();

  // Restore the notification queue internal flag
  fnRunner_->stopConsuming();
  fnRunner_->startConsumingInternal(this, queue_.get());

