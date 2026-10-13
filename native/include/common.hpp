// Shared helpers for the optional compiled core of ANYgeometry.
//
// Built against the CPython stable ABI (Py_LIMITED_API 3.11): one abi3 wheel
// per platform serves every CPython >= 3.11. Data crosses the boundary only as
// C-contiguous float64 buffers prepared and allocated by anygeometry.backend;
// the core holds no global mutable state.
#pragma once

#if defined(__FAST_MATH__)
#error "anygeometry._core must not be built with fast-math"
#endif

#ifndef Py_LIMITED_API
#error "anygeometry._core is built against the CPython stable ABI"
#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstddef>
#include <cstring>

namespace anygeometry_core {

// A read-only or writable view of a C-contiguous float64 buffer.
struct DoubleBuffer {
    Py_buffer view{};
    bool acquired = false;

    ~DoubleBuffer() {
        if (acquired) {
            PyBuffer_Release(&view);
        }
    }

    // Acquire ``object`` as contiguous doubles; sets a Python error on failure.
    bool acquire(PyObject* object, bool writable, const char* name) {
        int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(object, &view, flags) != 0) {
            return false;
        }
        acquired = true;
        if (view.itemsize != static_cast<Py_ssize_t>(sizeof(double)) || view.format == nullptr
                || !(std::strcmp(view.format, "d") == 0 || std::strcmp(view.format, "<d") == 0
                     || std::strcmp(view.format, "=d") == 0)) {
            PyErr_Format(PyExc_TypeError, "%s must be a float64 buffer", name);
            return false;
        }
        return true;
    }

    Py_ssize_t size() const { return view.len / static_cast<Py_ssize_t>(sizeof(double)); }
    const double* data() const { return static_cast<const double*>(view.buf); }
    double* mutable_data() { return static_cast<double*>(view.buf); }
};

}  // namespace anygeometry_core
