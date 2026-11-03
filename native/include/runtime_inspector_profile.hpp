#ifndef ANYGEOMETRY_RUNTIME_INSPECTOR_PROFILE_HPP
#define ANYGEOMETRY_RUNTIME_INSPECTOR_PROFILE_HPP
// Prospective inspector guard, copied from the adopted profile, never regenerated.
#define AG_RUNTIME_VERSION "3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)]"
#define AG_RUNTIME_SHA256 "7ffe3ab11342db03cc18b026f27485c0f74f5bf10f239f1f6573394900596394"
#define AG_RUNTIME_BYTES 6760792ULL
#define AG_HEADERS_SHA256 "70647ed05bf469f9d5fd21cb6fe3733c14c245c04c340d947428f72ce0442d72"
#define AG_PROFILE_SHA256 "23754f3aa260f172b0f0344e86fc98b3ef63940a59129b8751764d3f13e9c579"
#define AG_HEADER_COUNT 283
#ifndef AG_INSPECTOR_HEADERS_VALIDATED
#error "Build this inspector through the exact-header profile validator"
#endif
#if defined(Py_LIMITED_API) || defined(Py_GIL_DISABLED) || defined(Py_DEBUG) || defined(Py_TRACE_REFS) || defined(_DEBUG)
#error "The passive inspector requires non-limited release GIL CPython"
#endif
#if !defined(_WIN64) || !defined(_M_X64) || !defined(_MSC_VER) || _MSC_VER != 1950
#error "The passive inspector requires its declared Windows x64 MSVC1950 build"
#endif
#if !defined(_MSVC_LANG) || _MSVC_LANG != 201703L
#error "The passive inspector requires the declared C++17 layout configuration"
#endif
#endif
