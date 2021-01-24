import os
import sys
import sysconfig

if "--inplace" in sys.argv:
    from distutils.core import setup, find_packages
    from distutils.extension import Extension
else:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension


def config_cython():
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize
        ret = []
        path = "xflowrl/_cython"
        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(Extension(
                "xflowrl.%s" % fn[:-4],
                ["%s/%s" % (path, fn)],
                include_dirs=[".", "taso_ext/src", "/usr/local/cuda/include"],
                libraries=["taso_runtime", "taso_rl"],
                extra_compile_args=["-DUSE_CUDNN", "-std=c++11"],
                extra_link_args=[],
                language="c++"))
        return cythonize(ret, compiler_directives={"language_level" : 3})
    except ImportError:
        print("WARNING: cython is not installed!!!")
        return []


setup(
    name='xflowrl',
    version='0.1',
    description='A graph neural network agent.',
    url='',
    author='Kai Fricke, Michael Schaarschmidt',
    author_email='',
    license='',
    packages=[package for package in find_packages() if package.startswith('xflowrl')],
    ext_modules=config_cython(),
    install_requires=[
        'gast==0.3.3',
        'tensorflow==2.3.2',
        'dm-sonnet==2.0.0',
        'tensorflow-probability',
        'graph-nets',
        'gym'
    ]
)
