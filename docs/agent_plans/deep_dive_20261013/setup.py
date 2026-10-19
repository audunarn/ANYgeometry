from setuptools import setup, Extension
setup(name="kern", ext_modules=[Extension("kern", ["kern.c"], extra_compile_args=["/O2", "/fp:precise"])])
