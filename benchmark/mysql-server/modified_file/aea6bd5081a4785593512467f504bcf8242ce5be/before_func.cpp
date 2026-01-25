int
CPCD::findUniqueId() {
  int id;
  bool ok = false;
  m_processes.lock();
  
  while(!ok) {
    ok = true;
    id = rand() % 8192; /* Don't want so big numbers */

    if(id == 0)
      ok = false;

    for(unsigned i = 0; i<m_processes.size(); i++) {
      if(m_processes[i]->m_id == id)
	ok = false;
    }
  }
  m_processes.unlock();
  return id;
}