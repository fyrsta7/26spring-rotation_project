    , m_user_agent(MUST(String::from_utf8(default_user_agent)))
    , m_platform(MUST(String::from_utf8(default_platform)))
{
}

void ResourceLoader::prefetch_dns(URL::URL const& url)
{
    if (url.scheme().is_one_of("file"sv, "data"sv))
        return;

    if (ContentFilter::the().is_filtered(url)) {
        dbgln("ResourceLoader: Refusing to prefetch DNS for '{}': \033[31;1mURL was filtered\033[0m", url);
