# Writes OUT (translator_fingerprint.h): a SHA-256 over the translator's
# sources (src/gcn, every .cpp and .h, by name and content). The translation
# cache (src/host/translation_cache.cpp) keeps it in its file and starts empty
# when the running build's differs: a translator change throws away the
# translations of the one before. Rewritten only when the text changes, like
# bbhost_version.h, so nothing rebuilds needlessly.
file(GLOB sources ${SRC}/src/gcn/*.cpp ${SRC}/src/gcn/*.h)
list(SORT sources)
set(all "")
foreach(f ${sources})
  file(SHA256 ${f} h)
  get_filename_component(n ${f} NAME)
  string(APPEND all "${n}:${h};")
endforeach()
string(SHA256 fp "${all}")
string(SUBSTRING "${fp}" 0 32 fp)
set(content "#pragma once\n#define BBHOST_TRANSLATOR_FINGERPRINT \"${fp}\"\n")
set(old "")
if(EXISTS ${OUT})
  file(READ ${OUT} old)
endif()
if(NOT old STREQUAL content)
  file(WRITE ${OUT} "${content}")
endif()
