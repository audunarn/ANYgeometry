#define _USE_MATH_DEFINES
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
/* cylinder frame: o[3], a[3] axis, r[3] radial, c[3] circumferential, start, sweep, height */
typedef struct { double o[3], a[3], r[3], c[3], start, sweep, height; } Cyl;
static int parse_cyl(PyObject *t, Cyl *k) {
    double *p = (double *)k;
    if (!PyTuple_Check(t) || PyTuple_GET_SIZE(t) != 15) { PyErr_SetString(PyExc_TypeError, "cyl tuple"); return 0; }
    for (int i = 0; i < 15; i++) { p[i] = PyFloat_AsDouble(PyTuple_GET_ITEM(t, i)); }
    return !PyErr_Occurred();
}
static void uv(const Cyl *k, const double *x, double *u, double *v) {
    double d[3] = {x[0]-k->o[0], x[1]-k->o[1], x[2]-k->o[2]};
    double ax = d[0]*k->a[0]+d[1]*k->a[1]+d[2]*k->a[2];
    double q[3] = {d[0]-ax*k->a[0], d[1]-ax*k->a[1], d[2]-ax*k->a[2]};
    double ang = atan2(q[0]*k->c[0]+q[1]*k->c[1]+q[2]*k->c[2], q[0]*k->r[0]+q[1]*k->r[1]+q[2]*k->r[2]);
    double raw = ang - k->start, tw = 2.0*M_PI, delta;
    if (k->sweep > 0) { delta = fmod(raw, tw); if (delta < 0) delta += tw; }
    else { delta = -fmod(-raw, tw); if (delta > 0) delta -= tw; }
    *u = delta / k->sweep; *v = ax / k->height;
}
static PyObject *one(PyObject *s, PyObject *args) {
    PyObject *t; double x[3]; Cyl k;
    if (!PyArg_ParseTuple(args, "O(ddd)", &t, &x[0], &x[1], &x[2]) || !parse_cyl(t, &k)) return NULL;
    double u, v; uv(&k, x, &u, &v); return Py_BuildValue("(dd)", u, v);
}
static PyObject *many(PyObject *s, PyObject *args) {
    PyObject *t; Py_buffer in, out; Cyl k;
    if (!PyArg_ParseTuple(args, "Oy*w*", &t, &in, &out)) return NULL;
    if (!parse_cyl(t, &k)) { PyBuffer_Release(&in); PyBuffer_Release(&out); return NULL; }
    Py_ssize_t n = in.len / (3 * sizeof(double)); const double *x = in.buf; double *o = out.buf;
    for (Py_ssize_t i = 0; i < n; i++) uv(&k, x + 3*i, o + 2*i, o + 2*i + 1);
    PyBuffer_Release(&in); PyBuffer_Release(&out); Py_RETURN_NONE;
}
static PyMethodDef M[] = {{"one", one, METH_VARARGS, ""}, {"many", many, METH_VARARGS, ""}, {0}};
static struct PyModuleDef D = {PyModuleDef_HEAD_INIT, "kern", 0, -1, M};
PyMODINIT_FUNC PyInit_kern(void) { return PyModule_Create(&D); }
