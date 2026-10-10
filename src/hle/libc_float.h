#pragma once

#include <cstdlib>
#include <string>

namespace bb::libc {

inline double wide_stod(const char16_t* text, char16_t** end) {
    if (!text) return 0;
    std::string narrow;
    for (const char16_t* p = text; *p; ++p) {
        // C-locale floating syntax is ASCII. A non-ASCII code unit must
        // remain a terminator, not alias an ASCII digit after truncation.
        narrow.push_back(*p < 128 ? static_cast<char>(*p) : '\x7f');
    }
    char* tail = nullptr;
    const double value = std::strtod(narrow.c_str(), &tail);
    if (end) *end = const_cast<char16_t*>(text) + (tail - narrow.c_str());
    return value;
}

}  // namespace bb::libc
