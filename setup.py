#!/usr/bin/env python

import os
from os.path import dirname, isdir, join
import re
import shutil
import sys
from distutils.command.clean import clean as clean_command
from subprocess import CalledProcessError, check_output

from setuptools import Extension, setup
from setuptools.command.install import install as install_command
from setuptools.command.test import test as test_command
from Cython.Build import cythonize  # isort: skip This prevent a build bug


# Global variables to control generation of optional Cython-compiled library core module
BUILD_FROM_CYTHON = False
BUILD_FROM_SOURCE = False


def read(file_name):
    return open(os.path.join(os.path.dirname(__file__), file_name)).read()


# Implemented from http://stackoverflow.com/a/41110107
def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + "/__init__.py").read())
    return result.group(1)


class InstallCommand(install_command):
    """Overrides pip install command to control generation of optional Cython-compiled library core module"""

    user_options = install_command.user_options + [
        ("use-cython", None, "Cythonize and compile nurbs.core"),
        ("use-source", None, "Compile nurbs.core from the source files"),
        ("core-only", None, "Compile and install nurbs.core only"),
    ]

    def initialize_options(self):
        install_command.initialize_options(self)
        self.use_cython = 0
        self.use_source = 0

    def finalize_options(self):
        install_command.finalize_options(self)

    def run(self):
        global BUILD_FROM_CYTHON, BUILD_FROM_SOURCE
        BUILD_FROM_CYTHON = True if self.use_cython > 0 else False
        BUILD_FROM_SOURCE = True if self.use_source > 0 else False
        install_command.run(self)


# Reference: https://docs.pytest.org/en/latest/goodpractices.html
class PyTest(test_command):
    """Allows test command to call py.test"""

    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        test_command.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex

        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


class SetuptoolsClean(clean_command):
    """Cleans Cython-generated source files and setuptools-generated directories"""

    def run(self):
        # Call parent method
        clean_command.run(self)

        # Clean setuptools-generated directories
        st_dirs = ["dist", "build", "nurbs.egg-info", "nurbs.core.egg-info"]

        print("Removing setuptools-generated directories")
        for d in st_dirs:
            d_path = os.path.join(os.path.dirname(__file__), d)
            shutil.rmtree(d_path, ignore_errors=True)

        # Find list of files with .c extension
        flist_c, flist_c_path = read_files("nurbs", ".c")

        # Clean files with .c extensions
        if flist_c_path:
            print("Removing Cython-generated source files with .c extension")
            for f in flist_c_path:
                f_path = os.path.join(os.path.dirname(__file__), f)
                os.unlink(f_path)

        # Find list of files with .cpp extension
        flist_cpp, flist_cpp_path = read_files("nurbs", ".cpp")

        # Clean files with .cpp extensions
        if flist_cpp_path:
            print("Removing Cython-generated source files with .cpp extension")
            for f in flist_cpp_path:
                f_path = os.path.join(os.path.dirname(__file__), f)
                os.unlink(f_path)


def read_files(project, ext):
    """Reads files inside the input project directory."""
    project_path = os.path.join(os.path.dirname(__file__), project)
    file_list = os.listdir(project_path)
    flist = []
    flist_path = []
    for f in file_list:
        f_path = os.path.join(project_path, f)
        if os.path.isfile(f_path) and f.endswith(ext) and f != "__init__.py":
            flist.append(f.split(".")[0])
            flist_path.append(f_path)
    return flist, flist_path


def copy_files(src, ext, dst):
    """Copies files with extensions "ext" from "src" to "dst" directory."""
    src_path = os.path.join(os.path.dirname(__file__), src)
    dst_path = os.path.join(os.path.dirname(__file__), dst)
    file_list = os.listdir(src_path)
    for f in file_list:
        if f == "__init__.py":
            continue
        f_path = os.path.join(src_path, f)
        if os.path.isfile(f_path) and f.endswith(ext):
            shutil.copy(f_path, dst_path)


def make_dir(project):
    """Creates the project directory for compiled modules."""
    project_path = os.path.join(os.path.dirname(__file__), project)
    # Delete the directory and the files inside it
    if os.path.exists(project_path):
        shutil.rmtree(project_path)
    # Create the directory
    os.mkdir(project_path)
    # We need a __init__.py file inside the directory
    with open(os.path.join(project_path, "__init__.py"), "w") as fp:
        fp.write('__version__ = "' + str(get_property("__version__", "nurbs")) + '"\n')
        fp.write('__author__ = "' + str(get_property("__author__", "nurbs")) + '"\n')
        fp.write('__license__ = "' + str(get_property("__license__", "nurbs")) + '"\n')


def in_argv(arg_list):
    """Checks if any of the elements of the input list is in sys.argv array."""
    for arg in sys.argv:
        for parg in arg_list:
            if parg == arg or arg.startswith(parg):
                return True
    return False


def readme():
    with open("README.md") as f:
        return f.read()


tag_re = re.compile(r"\btag: %s([0-9][^,]*)\b")
version_re = re.compile("^Version: (.+)$", re.M)


def version_from_git_describe(version):
    if version[0] == "v":
        version = version[1:]
    # PEP 440 compatibility
    number_commits_ahead = 0
    if "-" in version:
        version, number_commits_ahead, commit_hash = version.split("-")
        number_commits_ahead = int(number_commits_ahead)

    split_versions = version.split(".")
    if "post" in split_versions[-1]:
        suffix = split_versions[-1]
        split_versions = split_versions[:-1]
    else:
        suffix = None
    for pre_release_segment in ["a", "b", "rc"]:
        if pre_release_segment in split_versions[-1]:
            if number_commits_ahead > 0:
                split_versions[-1] = str(split_versions[-1].split(pre_release_segment)[0])
                if len(split_versions) == 2:
                    split_versions.append("0")
                if len(split_versions) == 1:
                    split_versions.extend(["0", "0"])
                split_versions[-1] = str(int(split_versions[-1]) + 1)
                future_version = ".".join(split_versions)
                return "{}.dev{}+{}".format(future_version, number_commits_ahead, commit_hash)
            else:
                return ".".join(split_versions)
    if number_commits_ahead > 0:
        if len(split_versions) == 2:
            split_versions.append("0")
        if len(split_versions) == 1:
            split_versions.extend(["0", "0"])
        split_versions[-1] = str(int(split_versions[-1]) + 1)
        split_versions = ".".join(split_versions)
        return "{}.dev{}+{}".format(split_versions, number_commits_ahead, commit_hash)
    else:
        if suffix is not None:
            split_versions.append(suffix)
        return ".".join(split_versions)


# Just testing if get_version works well
assert version_from_git_describe("v0.1.7.post2") == "0.1.7.post2"
assert version_from_git_describe("v0.0.1-25-gaf0bf53") == "0.0.2.dev25+gaf0bf53"
assert version_from_git_describe("v0.1-15-zsdgaz") == "0.1.1.dev15+zsdgaz"
assert version_from_git_describe("v1") == "1"
assert version_from_git_describe("v1-3-aqsfjbo") == "1.0.1.dev3+aqsfjbo"
assert version_from_git_describe("v0.13rc0") == "0.13rc0"


def get_version():
    # Return the version if it has been injected into the file by git-archive
    version = tag_re.search("$Format:%D$")
    if version:
        return version.group(1)
    d = dirname(__file__)

    if isdir(join(d, ".git")):
        cmd = "git describe --tags"
        try:
            version = check_output(cmd.split()).decode().strip()[:]
        except CalledProcessError:
            raise RuntimeError("Unable to get version number from git tags")
        version = version_from_git_describe(version)
    else:
        # Extract the version from the PKG-INFO file.
        with open(join(d, "PKG-INFO")) as f:
            version = version_re.search(f.read()).group(1)

    # branch = get_branch()
    # if branch and branch != 'master':
    #     branch = re.sub('[^A-Za-z0-9]+', '', branch)
    #     if '+' in version:
    #         version += f'{branch}'
    #     else:
    #         version += f'+{branch}'

    return version


def get_branch():
    if isdir(join(dirname(__file__), ".git")):
        cmd = "git branch --show-current"
        try:
            return check_output(cmd.split()).decode().strip()[:]
        except CalledProcessError:
            pass

    return None


# Define setup.py commands to activate Cython compilation
possible_cmds = ["install", "build", "bdist"]

# Use geomdl.core package only
if "--core-only" in sys.argv:
    package_name = "nurbs.core"
    package_dir = "nurbs/core"
    packages = []
    sys.argv.remove("--core-only")
    sys.argv.append("--use-cython")
else:
    package_name = package_dir = "nurbs"
    packages = ["nurbs", "nurbs.visualization"]

# geomdl.core compilation
# Ref: https://gist.github.com/ctokheim/6c34dc1d672afca0676a

# Use already Cythonized C code
if in_argv(possible_cmds) and "--use-source" in sys.argv:
    BUILD_FROM_SOURCE = True
    sys.argv.remove("--use-source")

# Use Cython to (re)generate C code (overrides "--use-source")
if in_argv(possible_cmds) and "--use-cython" in sys.argv:
    # Try to import Cython
    try:
        from Cython.Build import cythonize
    except ImportError:
        raise ImportError("Cython is required for this step. Please install it via 'pip install cython'")

    BUILD_FROM_CYTHON = True
    BUILD_FROM_SOURCE = False
    sys.argv.remove("--use-cython")

# We don't want to include any compiled files with the distribution
ext_modules = []

if BUILD_FROM_CYTHON or BUILD_FROM_SOURCE:
    # Choose the file extension
    file_ext = ".py" if BUILD_FROM_CYTHON else ".c"

    # Create Cython-compiled module directory
    make_dir("nurbs/core")

    # Create extensions
    optional_extensions = []
    fnames, fnames_path = read_files("nurbs", file_ext)
    for fname, fpath in zip(fnames, fnames_path):
        temp = Extension("nurbs.core." + str(fname), sources=[fpath])
        optional_extensions.append(temp)

    # Call Cython when "python setup.py build_ext --use-cython" is executed
    if BUILD_FROM_CYTHON:
        ext_modules = cythonize(optional_extensions, compiler_directives={"language_level": sys.version_info[0]})

    # Compile from C source when "python setup.py build_ext --use-source" is executed
    if BUILD_FROM_SOURCE:
        ext_modules = optional_extensions

    # Add Cython-compiled module to the packages list
    packages.append("nurbs.core")

# Input for setuptools.setup
data = dict(
    name=package_name,
    version=get_version(),
    description=get_property("__description__", package_dir),
    long_description=read("DESCRIPTION.md"),
    long_description_content_type="text/markdown",
    license=get_property("__license__", package_dir),
    author=get_property("__author__", package_dir),
    author_email="root@dessia.tech",
    url="https://github.com/Dessia-tech/nurbs",
    keywords=get_property("__keywords__", package_dir),
    packages=packages,
    install_requires=[],
    tests_require=["pytest>=3.6.0"],
    cmdclass={"install": InstallCommand, "test": PyTest, "clean": SetuptoolsClean},
    ext_modules=cythonize(
        [
            "nurbs/evaluators.pyx",
            "nurbs/helpers.pyx",
            "nurbs/linalg.pyx",
        ],
        annotate=True
    ),
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    project_urls={
        "Documentation": "https://github.com/Dessia-tech/nurbs/doc",
        "Source": "https://github.com/Dessia-tech/nurbs",
        "Tracker": "https://github.com/Dessia-tech/nurbs/issues",
    },
)


if __name__ == "__main__":
    setup(**data)
