// Define destructor here so we can forward declare `Impl` in client_session.h.
// If we define a dtor in the header file or use the default dtor,
// unique_ptr<Impl> needs the complete type.
ClientSession::~ClientSession() {}
