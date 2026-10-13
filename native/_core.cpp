// anygeometry._core (native/_core.cpp): optional compiled kernels (stable ABI, C++17).
//
// Every exported kernel has a pure-Python reference with the same signature in
// anygeometry.reference; anygeometry.backend selects one of the two at import
// and the differential tests compare them. Kernels take C-contiguous float64
// buffers and write into caller-allocated outputs; they return None or raise.
#include "common.hpp"
#include "bspline.hpp"

#include <cmath>

namespace {

using anygeometry_core::DoubleBuffer;

// plane_evaluate(frame, uv, out): out[i] = origin + u_i * u_vector + v_i * v_vector.
// ``frame`` holds origin, u_vector and v_vector (9 doubles); ``uv`` is (n, 2);
// ``out`` is (n, 3). Same operation order as the reference.
PyObject* plane_evaluate(PyObject*, PyObject* args) {
    PyObject *frame_object, *uv_object, *out_object;
    if (!PyArg_ParseTuple(args, "OOO", &frame_object, &uv_object, &out_object)) {
        return nullptr;
    }
    DoubleBuffer frame, uv, out;
    if (!frame.acquire(frame_object, false, "frame") || !uv.acquire(uv_object, false, "uv")
            || !out.acquire(out_object, true, "out")) {
        return nullptr;
    }
    if (frame.size() != 9 || uv.size() % 2 != 0 || out.size() != uv.size() / 2 * 3) {
        PyErr_SetString(PyExc_ValueError, "plane_evaluate: inconsistent buffer sizes");
        return nullptr;
    }
    const double* f = frame.data();
    const double* p = uv.data();
    double* o = out.mutable_data();
    const Py_ssize_t count = uv.size() / 2;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < count; ++i) {
        const double u = p[2 * i], v = p[2 * i + 1];
        for (int k = 0; k < 3; ++k) {
            o[3 * i + k] = (f[k] + u * f[3 + k]) + v * f[6 + k];
        }
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

// bspline_evaluate(degree, knots, controls, weights, t, order, out): curve points
// and derivatives (reference.bspline.bspline_evaluate). ``controls`` is (n, dim)
// with dim <= 4, ``weights`` is (n,) or empty, ``out`` is (m, order+1, dim).
PyObject* bspline_evaluate(PyObject*, PyObject* args) {
    int degree, order;
    PyObject *knots_object, *controls_object, *weights_object, *t_object, *out_object;
    if (!PyArg_ParseTuple(args, "iOOOOiO", &degree, &knots_object, &controls_object, &weights_object, &t_object,
                          &order, &out_object)) {
        return nullptr;
    }
    DoubleBuffer knots, controls, weights, t, out;
    if (!knots.acquire(knots_object, false, "knots") || !controls.acquire(controls_object, false, "controls")
            || !weights.acquire(weights_object, false, "weights") || !t.acquire(t_object, false, "t")
            || !out.acquire(out_object, true, "out")) {
        return nullptr;
    }
    using namespace anygeometry_core;
    if (degree < 1 || degree > kMaxDegree || order < 0 || order > kMaxOrder || controls.ndim() != 2) {
        PyErr_SetString(PyExc_ValueError, "bspline_evaluate: unsupported degree, order or control shape");
        return nullptr;
    }
    const Py_ssize_t count = controls.extent(0), dim = controls.extent(1), m = t.size();
    if (dim < 1 || dim > kMaxDimension || count < degree + 1 || knots.size() != count + degree + 1
            || (weights.size() != 0 && weights.size() != count) || out.size() != m * (order + 1) * dim) {
        PyErr_SetString(PyExc_ValueError, "bspline_evaluate: inconsistent buffer sizes");
        return nullptr;
    }
    const double* k = knots.data();
    const double lower = k[degree], upper = k[count];
    for (Py_ssize_t i = 0; i < m; ++i) {
        const double value = t.data()[i];
        if (!(value >= lower && value <= upper)) {
            PyErr_SetString(PyExc_ValueError, "bspline_evaluate: parameter outside the knot domain");
            return nullptr;
        }
    }
    const double* w = weights.size() == 0 ? nullptr : weights.data();
    double* o = out.mutable_data();
    const int c = static_cast<int>(count), d = static_cast<int>(dim);
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < m; ++i) {
        evaluate_point(degree, k, c, controls.data(), d, w, t.data()[i], order, o + i * (order + 1) * dim);
    }
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

// build_info() -> dict describing how this module was compiled.
PyObject* build_info(PyObject*, PyObject*) {
#if defined(_MSC_VER)
    const char* compiler = "msvc";
    const long compiler_version = _MSC_VER;
#elif defined(__clang__)
    const char* compiler = "clang";
    const long compiler_version = __clang_major__;
#elif defined(__GNUC__)
    const char* compiler = "gcc";
    const long compiler_version = __GNUC__;
#else
    const char* compiler = "unknown";
    const long compiler_version = 0;
#endif
    return Py_BuildValue("{s:s,s:l,s:s,s:i,s:i}", "compiler", compiler, "compiler_version", compiler_version,
                         "abi", "abi3", "limited_api", static_cast<int>(Py_LIMITED_API),
                         "core_api", 1);
}

PyMethodDef methods[] = {
    {"plane_evaluate", plane_evaluate, METH_VARARGS, "Evaluate plane points for (n, 2) parameters."},
    {"bspline_evaluate", bspline_evaluate, METH_VARARGS, "B-spline/NURBS curve points and derivatives."},
    {"build_info", build_info, METH_NOARGS, "How this module was compiled."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_core", "Optional compiled kernels for ANYgeometry.", -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit__core(void) { return PyModule_Create(&module); }
