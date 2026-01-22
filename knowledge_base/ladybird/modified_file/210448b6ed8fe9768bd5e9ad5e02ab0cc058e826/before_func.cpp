#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {

// https://pubs.opengroup.org/onlinepubs/9699919799/functions/strspn.html
size_t strspn(char const* s, char const* accept)
{
    char const* p = s;
cont:
    char ch = *p++;
    char ac;
    for (char const* ap = accept; (ac = *ap++) != '\0';) {
