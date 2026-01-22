void ThreadPool::schedule(Task task){
    boostlock lock(_mutex);

    _tasksRemaining++;

    if (!_freeWorkers.empty()){
        _freeWorkers.front()->set_task(task);
        _freeWorkers.pop_front();
    }else{
        _tasks.push_back(task);
    }
}

// should only be called by a worker from the worker thread
void ThreadPool::task_done(Worker* worker){
