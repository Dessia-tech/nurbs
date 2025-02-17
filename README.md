# Nurbs


## Introduction

nurbs is a fork of geomdl.

NURBS-Python (geomdl) is a pure Python, self-contained, object-oriented B-Spline and NURBS spline library for Python
versions 2.7.x, 3.4.x and later.

The `following article <https://doi.org/10.1016/j.softx.2018.12.005>`_ outlines the design and features of NURBS-Python
(geomdl). I would be glad if you would cite it if you have used NURBS-Python (geomdl) in your research::

    @article{bingol2019geomdl,
      title={{NURBS-Python}: An open-source object-oriented {NURBS} modeling framework in {Python}},
      author={Bingol, Onur Rauf and Krishnamurthy, Adarsh},
      journal={{SoftwareX}},
      volume={9},
      pages={85--94},
      year={2019},
      publisher={Elsevier},
      doi={https://doi.org/10.1016/j.softx.2018.12.005}
    }

## Features

NURBS-Python (geomdl) provides convenient data structures and highly customizable API for rational and non-rational
splines along with the efficient and extensible implementations of the following algorithms:

* Spline evaluation
* Derivative evaluation
* Knot insertion
* Knot removal
* Knot vector refinement
* Degree elevation
* Degree reduction
* Curve and surface fitting via interpolation and least squares approximation

NURBS-Python (geomdl) also provides customizable visualization and animation options via Matplotlib, Plotly and VTK
libraries. Please refer to the `documentation <http://nurbs-python.readthedocs.io/>`_ for more details.

## Installation

The easiest way to install NURBS-Python (geomdl) is using ``pip``:

.. code-block:: console

    $ pip install --user geomdl

It is also possible to install NURBS-Python (geomdl) using ``conda``:

.. code-block:: console

    $ conda install -c orbingol geomdl

Please refer to the `Installation and Testing <http://nurbs-python.readthedocs.io/en/latest/install.html>`_ section
of the documentation for alternative installation methods.

## Examples and Documentation

* **Examples**: https://github.com/orbingol/NURBS-Python_Examples
* **Documentation**: http://nurbs-python.readthedocs.io/
* **Wiki**: https://github.com/orbingol/NURBS-Python/wiki

## Extra Modules

* **Command-line application**: https://github.com/orbingol/geomdl-cli
* **Shapes module**: https://github.com/orbingol/geomdl-shapes
* **Rhino importer/exporter**: https://github.com/orbingol/rw3dm
* **ACIS exporter**: https://github.com/orbingol/rwsat

## Author

* Onur R. Bingol (`@orbingol <https://github.com/orbingol>`_), original author of geomdl
* Dessia Technologies


## Acknowledgments

Please see `CONTRIBUTORS.rst <CONTRIBUTORS.rst>`_ file for the acknowledgements.

## License

NURBS-Python (geomdl) is licensed under the terms of `MIT License <LICENSE>`_ and it contains the following modules:

* ``six`` is licensed under the terms of `MIT License <https://github.com/benjaminp/six/blob/1.12.0/LICENSE>`_
* ``backports.functools_lru_cache`` is licensed under the terms of `MIT License <https://github.com/jaraco/backports.functools_lru_cache/blob/1.5/LICENSE>`_
