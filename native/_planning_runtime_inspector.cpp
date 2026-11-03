// Exact-profile passive observations. No codec lookup/search invocation.
// Python allocations and final decrefs may run GC/finalizers.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include <Python.h>
#include "internal/pycore_interp_structs.h"
#include "internal/pycore_runtime_structs.h"
#include "runtime_inspector_profile.hpp"
#include <cstddef>
#include <cstring>
#include <new>

static_assert(sizeof(void*) == 8 && sizeof(long) == 4, "Windows x64 layout");
static_assert(offsetof(codecs_state, search_path) == 0 &&
              offsetof(codecs_state, search_cache) == 8 &&
              offsetof(codecs_state, error_registry) == 16 &&
              offsetof(codecs_state, initialized) == 24 && sizeof(codecs_state) == 32,
              "declared GIL codec layout");
using AuditHookState=decltype(_PyRuntimeState::audit_hooks);
static_assert(sizeof(PyMutex)==1 && offsetof(AuditHookState,mutex)==0 &&
              offsetof(AuditHookState,head)==8 && sizeof(AuditHookState)==16,
              "declared runtime audit mutex/head layout");
static const char* keys[] = {"utf_8", "cp1252", "strict", "backslashreplace"};
static const char* capsule_name = "anygeometry.codec_registry.v1";
// Failed closes retain uncertain identities and permanently refuse acquisition.
// Never blindly retry an uncertain native close.
static HANDLE uncertain_file = nullptr;
static BCRYPT_ALG_HANDLE uncertain_algorithm = nullptr;
static BCRYPT_HASH_HANDLE uncertain_hash = nullptr;

// Native handles never cross a Python API boundary, including failure paths.
static bool runtime_image_matches() {
    if (uncertain_file || uncertain_algorithm || uncertain_hash) return false;
    HMODULE module = nullptr;
    wchar_t path[32768];
    if (!GetModuleHandleExW(0x6, reinterpret_cast<LPCWSTR>(&Py_GetVersion), &module)) return false;
    DWORD length = GetModuleFileNameW(module, path, 32768);
    if (!length || length >= 32768) return false;
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    LARGE_INTEGER size;
    bool ok = GetFileSizeEx(file, &size) && size.QuadPart == AG_RUNTIME_BYTES;
    if (ok) ok = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0;
    if (ok) ok = BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) >= 0;
    unsigned char buffer[65536], digest[32];
    unsigned long long total = 0;
    while (ok) {
        DWORD count = 0;
        if (!ReadFile(file, buffer, sizeof(buffer), &count, nullptr)) { ok = false; break; }
        if (!count) break;
        total += count;
        if (total > AG_RUNTIME_BYTES || BCryptHashData(hash, buffer, count, 0) < 0) ok = false;
    }
    if (ok) ok = total == AG_RUNTIME_BYTES && BCryptFinishHash(hash, digest, sizeof(digest), 0) >= 0;
    if (hash && BCryptDestroyHash(hash) < 0) { uncertain_hash=hash; ok=false; }
    if (algorithm && BCryptCloseAlgorithmProvider(algorithm, 0) < 0) { uncertain_algorithm=algorithm; ok=false; }
    if (!CloseHandle(file)) { uncertain_file=file; ok=false; }
    if (!ok) return false;
    const char hex[] = "0123456789abcdef";
    char encoded[65];
    for (int i = 0; i < 32; ++i) { encoded[2*i] = hex[digest[i] >> 4]; encoded[2*i+1] = hex[digest[i] & 15]; }
    encoded[64] = 0;
    return std::strcmp(encoded, AG_RUNTIME_SHA256) == 0;
}

static bool guard() {
    if (std::strcmp(Py_GetVersion(), AG_RUNTIME_VERSION) || !runtime_image_matches()) {
        PyErr_SetString(PyExc_RuntimeError, uncertain_file || uncertain_algorithm || uncertain_hash
            ? "passive inspector native cleanup uncertain; permanently poisoned"
            : "passive inspector runtime profile mismatch");
        return false;
    }
    return true;
}

struct Observation {
    PyInterpreterState* interpreter = nullptr;
    bool audit_hooks_absent = false;
    PyObject* stores[3] = {};
    PyObject* path[64] = {};
    Py_ssize_t path_size = 0;
    PyObject* selected[4] = {}; // nullptr means missing; Py_None means present.
};

static bool audits_absent(PyInterpreterState* interpreter) {
    // CPython v3.14.2 sysmodule.c:472 uses this mutex for head insertion.
    // Public Lock's slowpath uses DETACH only, not HANDLE_SIGNALS. No Python
    // API/allocations/borrowed Python references while holding the mutex.
    auto* runtime=interpreter->runtime;
    PyMutex_Lock(&runtime->audit_hooks.mutex);
    const bool native_absent=runtime->audit_hooks.head==nullptr;
    PyMutex_Unlock(&runtime->audit_hooks.mutex);
    // Lock reacquires the GIL before returning. Read interpreter-local state
    // after unlocking, without traversing or retaining any hook contents.
    PyObject* hooks=interpreter->audit_hooks;
    return native_absent && (!hooks || (PyList_CheckExact(hooks) && PyList_GET_SIZE(hooks)==0));
}
static PyObject* audit_refusal() {
    PyErr_SetString(PyExc_RuntimeError,"passive inspector requires absent audit hooks");
    return nullptr;
}

static bool read_current(Observation& value) {
    value.interpreter = PyThreadState_GetInterpreter(PyThreadState_Get());
    value.audit_hooks_absent=audits_absent(value.interpreter);
    if (!value.audit_hooks_absent) return true;
    const codecs_state& state = value.interpreter->codecs;
    if (!state.initialized || !state.search_path || !state.search_cache || !state.error_registry ||
        !PyList_CheckExact(state.search_path) || !PyDict_CheckExact(state.search_cache) ||
        !PyDict_CheckExact(state.error_registry)) {
        PyErr_SetString(PyExc_RuntimeError, "unsupported codec registry containers"); return false;
    }
    value.stores[0] = state.search_path; value.stores[1] = state.search_cache; value.stores[2] = state.error_registry;
    value.path_size = PyList_GET_SIZE(state.search_path);
    if (value.path_size > 64 || PyDict_Size(state.search_cache) > 512 || PyDict_Size(state.error_registry) > 64) {
        PyErr_SetString(PyExc_RuntimeError, "codec registry observation cap exceeded"); return false;
    }
    for (Py_ssize_t i=0; i<value.path_size; ++i) value.path[i] = PyList_GET_ITEM(state.search_path, i);
    for (int store=1; store<3; ++store) {
        Py_ssize_t position=0; PyObject *key, *item;
        while (PyDict_Next(value.stores[store], &position, &key, &item)) {
            // Never hash or compare arbitrary Python keys, even unrelated ones.
            if (!PyUnicode_CheckExact(key)) {
                PyErr_SetString(PyExc_RuntimeError, "non-exact codec registry key"); return false;
            }
            for (int i=(store-1)*2; i<store*2; ++i) {
                int comparison=PyUnicode_CompareWithASCIIString(key, keys[i]);
                if (comparison == -1 && PyErr_Occurred()) return false;
                if (!comparison) value.selected[i]=item;
            }
        }
    }
    return true;
}

static bool same(const Observation& a, const Observation& b) {
    if (!a.audit_hooks_absent || !b.audit_hooks_absent) return false;
    if (a.interpreter != b.interpreter || a.path_size != b.path_size) return false;
    for (int i=0; i<3; ++i) if (a.stores[i] != b.stores[i]) return false;
    for (Py_ssize_t i=0; i<a.path_size; ++i) if (a.path[i] != b.path[i]) return false;
    for (int i=0; i<4; ++i) if (a.selected[i] != b.selected[i]) return false;
    return true;
}
static void retain(Observation& a) {
    for (auto value:a.stores) Py_XINCREF(value);
    for (Py_ssize_t i=0;i<a.path_size;++i) Py_XINCREF(a.path[i]);
    for (auto value:a.selected) Py_XINCREF(value);
}
static void release(Observation* a) {
    for (auto value:a->stores) Py_XDECREF(value);
    for (Py_ssize_t i=0;i<a->path_size;++i) Py_XDECREF(a->path[i]);
    for (auto value:a->selected) Py_XDECREF(value);
    delete a;
}
static void destroy(PyObject* capsule) {
    PyObject* error=PyErr_GetRaisedException();
    auto* value=static_cast<Observation*>(PyCapsule_GetPointer(capsule,capsule_name));
    if (value) release(value);
    PyErr_Clear();
    PyErr_SetRaisedException(error);
}
static PyObject* rows(const Observation& a, int start) {
    PyObject* result=PyTuple_New(2);
    if (!result) return nullptr;
    for (int j=0;j<2;++j) {
        int i=start+j;
        PyObject* row=Py_BuildValue("(sOO)",keys[i],a.selected[i]?Py_True:Py_False,a.selected[i]?a.selected[i]:Py_None);
        if (!row) { Py_DECREF(result); return nullptr; }
        PyTuple_SET_ITEM(result,j,row);
    }
    return result;
}
static PyObject* snapshot(PyObject*, PyObject*) {
    if (!guard()) return nullptr;
    auto* held=new(std::nothrow) Observation;
    if (!held) return PyErr_NoMemory();
    if (!read_current(*held)) { delete held; return nullptr; }
    if (!held->audit_hooks_absent) { delete held; return audit_refusal(); }
    retain(*held); // Before allocations which could run cyclic GC.
    PyObject* capsule=PyCapsule_New(held,capsule_name,destroy);
    if (!capsule) { release(held); return nullptr; }
    PyObject* path=PyTuple_New(held->path_size);
    if (path) for (Py_ssize_t i=0;i<held->path_size;++i) PyTuple_SET_ITEM(path,i,Py_NewRef(held->path[i]));
    PyObject* codecs=path?rows(*held,0):nullptr;
    PyObject* errors=codecs?rows(*held,2):nullptr;
    PyObject* result=errors?PyTuple_Pack(4,capsule,path,codecs,errors):nullptr;
    Py_XDECREF(path); Py_XDECREF(codecs); Py_XDECREF(errors);
    if (result) {
        Observation current;
        if (!read_current(current) || !same(*held,current)) {
            Py_DECREF(result); result=nullptr;
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError,"codec registry changed during observation");
        }
    }
    Py_DECREF(capsule);
    return result;
}
static PyObject* matches(PyObject*, PyObject* capsule) {
    if (!guard()) return nullptr;
    auto* held=static_cast<Observation*>(PyCapsule_GetPointer(capsule,capsule_name));
    if (!held) return nullptr;
    Observation current;
    if (!read_current(current)) return nullptr;
    return PyBool_FromLong(same(*held,current));
}
static PyObject* identity(PyObject*, PyObject*) {
    if (!guard()) return nullptr;
    auto* interpreter=PyThreadState_GetInterpreter(PyThreadState_Get());
    if (!audits_absent(interpreter)) return audit_refusal();
    PyObject* result=Py_BuildValue("{s:i,s:s,s:s,s:s,s:s,s:i,s:i,s:n,s:n,s:n,s:O,s:n,s:n,s:n,s:n}",
        "schema",2,"abi","cp314-cp314-win_amd64","runtime_sha256",AG_RUNTIME_SHA256,
        "headers_sha256",AG_HEADERS_SHA256,"profile_sha256",AG_PROFILE_SHA256,
        "runtime_msc_ver",1944,"helper_msc_ver",_MSC_VER,
        "interpreter_codecs_offset",static_cast<Py_ssize_t>(offsetof(PyInterpreterState,codecs)),
        "interpreter_size",static_cast<Py_ssize_t>(sizeof(PyInterpreterState)),
        "codecs_size",static_cast<Py_ssize_t>(sizeof(codecs_state)),
        "audit_hooks_absent",Py_True,
        "interpreter_audit_offset",static_cast<Py_ssize_t>(offsetof(PyInterpreterState,audit_hooks)),
        "runtime_audit_mutex_offset",static_cast<Py_ssize_t>(offsetof(_PyRuntimeState,audit_hooks)+offsetof(AuditHookState,mutex)),
        "runtime_audit_head_offset",static_cast<Py_ssize_t>(offsetof(_PyRuntimeState,audit_hooks)+offsetof(AuditHookState,head)),
        "runtime_size",static_cast<Py_ssize_t>(sizeof(_PyRuntimeState)));
    if (result && !audits_absent(interpreter)) { Py_DECREF(result); return audit_refusal(); }
    return result;
}
static PyMethodDef methods[]={{"snapshot",snapshot,METH_NOARGS,nullptr},{"matches",matches,METH_O,nullptr},
                             {"identity",identity,METH_NOARGS,nullptr},{nullptr,nullptr,0,nullptr}};
static PyModuleDef definition={PyModuleDef_HEAD_INIT,"_planning_runtime_inspector",nullptr,0,methods};
PyMODINIT_FUNC PyInit__planning_runtime_inspector() { return PyModule_Create(&definition); }
