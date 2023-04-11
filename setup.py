from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize


class BuildRvo2Ext(_build_ext):
    """Builds RVO2 before our module."""

    def run(self):
        # Build RVO2
        import os
        import os.path
        import subprocess

        build_dir = os.path.abspath('build/RVO2')
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
            subprocess.check_call(['cmake', '../..', '-DCMAKE_CXX_FLAGS=-fPIC'],
                                  cwd=build_dir)
        subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)

        _build_ext.run(self)


extensions = [
    Extension('rvo2', ['src/*.pyx'],
              include_dirs=['src'],
              libraries=['RVO'],
              library_dirs=['build/RVO2/src'],
              extra_compile_args=['-fPIC']),
]

setup(
    name="indoorca",
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': BuildRvo2Ext},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Cython',
        'Topic :: Games/Entertainment :: Simulation',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('test*',)),
    include_package_data=True,
    install_requires=[
        'numpy>=1.19.5',
        'Cython==0.21.1',
        'jupyter>=1.0.0',
        'jupytext>=1.2.0',
        'Pillow>=5.4.0',
        'networkx>=2.0',
        'opencv-python>=3.4.8',
        'matplotlib==3.3.4',
        'shapely==1.7.1',
        'scikit-image==0.17.2'
    ]
)
