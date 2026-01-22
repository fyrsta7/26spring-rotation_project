#include <AK/Badge.h>
#include <AK/HashTable.h>
#include <LibJS/Heap/Handle.h>
#include <LibJS/Heap/Heap.h>
#include <LibJS/Heap/HeapBlock.h>
#include <LibJS/Interpreter.h>
#include <LibJS/Runtime/Object.h>
#include <serenity.h>
#include <setjmp.h>
#include <stdio.h>

#define HEAP_DEBUG

namespace JS {

Heap::Heap(Interpreter& interpreter)
    : m_interpreter(interpreter)
{
