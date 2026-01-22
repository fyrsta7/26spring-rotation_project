QStringList misc::toStringList(const QList<bool> &l) {
  QStringList ret;
  foreach (const bool &b, l) {
    ret << (b ? "1" : "0");
  }
  return ret;
}

QList<int> misc::intListfromStringList(const QStringList &l) {
  QList<int> ret;
  foreach (const QString &s, l) {
    ret << s.toInt();
  }
  return ret;
}

QList<bool> misc::boolListfromStringList(const QStringList &l) {
  QList<bool> ret;
  foreach (const QString &s, l) {
    ret << (s=="1");
  }
  return ret;
}

bool misc::isUrl(const QString &s)
{
  const QString scheme = QUrl(s).scheme();
  QRegExp is_url("http[s]?|ftp", Qt::CaseInsensitive);
  return is_url.exactMatch(scheme);
}

QString misc::parseHtmlLinks(const QString &raw_text)
{
  QString result = raw_text;
  QRegExp reURL("(\\s|^)"                                     //start with whitespace or beginning of line
                "("
                "("                                      //case 1 -- URL with scheme
                "(http(s?))\\://"                    //start with scheme
                "([a-zA-Z0-9_-]+\\.)+"               //  domainpart.  at least one of these must exist
                "([a-zA-Z0-9\\?%=&/_\\.:#;-]+)"      //  everything to 1st non-URI char, must be at least one char after the previous dot (cannot use ".*" because it can be too greedy)
                ")"
                "|"
                "("                                     //case 2a -- no scheme, contains common TLD  example.com
                "([a-zA-Z0-9_-]+\\.)+"              //  domainpart.  at least one of these must exist
                "(?="                               //  must be followed by TLD
                "AERO|aero|"                  //N.B. assertions are non-capturing
                "ARPA|arpa|"
                "ASIA|asia|"
                "BIZ|biz|"
                "CAT|cat|"
                "COM|com|"
                "COOP|coop|"
                "EDU|edu|"
                "GOV|gov|"
                "INFO|info|"
                "INT|int|"
                "JOBS|jobs|"
                "MIL|mil|"
                "MOBI|mobi|"
