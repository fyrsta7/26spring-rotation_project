static inline bool bstr_equals(struct bstr str1, struct bstr str2)
{
    if (str1.len != str2.len)
        return false;

    return str1.start == str2.start || bstrcmp(str1, str2) == 0;
}