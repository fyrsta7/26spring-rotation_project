    {
        this->print(node->getSubNodes().at(i), offset);
    }
}
void    DOMParser::init                     ()
{
    this->root          = NULL;
    this->vlc_reader    = NULL;
}
void    DOMParser::print                    ()
{
    this->print(this->root, 0);
}
Profile DOMParser::getProfile               (dash::xml::Node *node)
{
    std::string profile = node->getAttributeValue("profiles");

    if(!profile.compare("urn:mpeg:mpegB:profile:dash:isoff-basic-on-demand:cm"))
        return dash::mpd::BasicCM;
