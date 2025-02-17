NURBS-Python v5.x Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|DOI|_ |PYPI|_ |ANACONDA|_

|TRAVISCI|_ |APPVEYOR|_ |CIRCLECI|_

Welcome to the **NURBS-Python (nurbs) v5.x** documentation!

NURBS-Python (nurbs) is a cross-platform (pure Python), object-oriented B-Spline and NURBS library.
It is compatible with Python versions 2.7.x, 3.4.x and later.
It supports rational and non-rational curves, surfaces and volumes.

NURBS-Python (nurbs) provides easy-to-use data structures for storing geometry descriptions
in addition to the fundamental and advanced evaluation algorithms.

This documentation is organized into a couple sections:

* :ref:`introduction`
* :ref:`using`
* :ref:`modules`

.. _introduction:

.. toctree::
    :maxdepth: 2
    :caption: Introduction

    introduction
    citing
    q_a
    contributing

.. _using:

.. toctree::
    :maxdepth: 2
    :caption: Using the Library

    install
    basics
    examples_repo
    load_save
    file_formats
    compatibility
    surface_generator
    knot_refinement
    fitting
    visualization
    visualization_splitting
    visualization_export

.. _modules:

.. toctree::
    :maxdepth: 3
    :caption: Modules

    modules
    modules_visualization
    modules_cli
    modules_shapes
    modules_rhino
    modules_acis


.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.815010.svg
.. _DOI: https://doi.org/10.5281/zenodo.815010

.. |PYPI| image:: https://img.shields.io/pypi/v/nurbs.svg
.. _PYPI: https://pypi.org/project/nurbs/

.. |ANACONDA| image:: https://anaconda.org/orbingol/nurbs/badges/version.svg
.. _ANACONDA: https://anaconda.org/orbingol/nurbs

.. |TRAVISCI| image:: https://travis-ci.org/orbingol/NURBS-Python.svg?branch=5.x
.. _TRAVISCI: https://travis-ci.org/orbingol/NURBS-Python

.. |APPVEYOR| image:: https://ci.appveyor.com/api/projects/status/github/orbingol/nurbs-python?branch=5.x&svg=true
.. _APPVEYOR: https://ci.appveyor.com/project/orbingol/nurbs-python

.. |CIRCLECI| image:: https://circleci.com/gh/orbingol/NURBS-Python/tree/5.x.svg?style=shield
.. _CIRCLECI: https://circleci.com/gh/orbingol/NURBS-Python/tree/5.x
