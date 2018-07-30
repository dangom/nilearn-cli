nilearn-cli
===========

-----
Convenient command line tools for `Nilearn <http://nilearn.github.io/>`_.
Currently provides three tools:

- connectome, which generates connectivity matrices from an input dataset.
- region_extraction, which generates masks from connected regions in a statistical map.
- surface_plot, which plots volumetric statistical maps into a surface and outputs a PNG.

.. contents:: **Table of Contents**
    :backlinks: none

Installation
------------

nilearn-cli is not yet distributed on `PyPI <https://pypi.org>`_ as a universal
wheel. To install it, clone the repo and run:

.. code-block:: bash

    $ python setup.py install

License
-------

nilearn-cli is distributed under the terms of both

- `MIT License <https://choosealicense.com/licenses/mit>`_
- `Apache License, Version 2.0 <https://choosealicense.com/licenses/apache-2.0>`_

at your option.
