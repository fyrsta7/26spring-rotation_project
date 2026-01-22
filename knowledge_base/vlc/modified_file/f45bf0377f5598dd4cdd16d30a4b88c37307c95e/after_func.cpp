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
