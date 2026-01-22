  loadingProcessList = false;
  m_processes.clear();
  m_monitor = NULL;
  m_monitor = new Monitor(this);
  m_procfile = "ndb_cpcd.db";
}

CPCD::~CPCD() {
  if(m_monitor != NULL) {
    delete m_monitor;
    m_monitor = NULL;
  }
}

int
CPCD::findUniqueId() {
  int id;
  bool ok = false;
  m_processes.lock();
  
  while(!ok) {
