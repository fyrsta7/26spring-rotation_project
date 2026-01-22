inline void spdlog::sinks::async_sink::_sink_it(const details::log_msg& msg)
{
    using namespace spdlog::details;
    _push_sentry();      
    _q.push(std::unique_ptr<log_msg>(new log_msg(msg)));
}
