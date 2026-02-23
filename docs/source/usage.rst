Usage
=====

Installation
------------

The best way to install at the moment is from PyPi into 
an isolated environment. If you are a conda user, the following
should work:

.. code-block:: console

    $ conda create -n test-release python=3.11
    $ conda activate test-release
    $ pip install pytomofilt
    $ ptf_reparam_filter_files --help

.. note:

    Further instructions to follow...

Filtering a model
-----------------

Assuming you have a bunch of files from a geodynamic model in directory "A"
and a tomographic model including the filter in a directory called "S12RTS" you can
generate a reparameterized and filtered representation of the geodynamic 
model with:

.. code-block:: console

    ptf_reparam_filter_files S12RTS A

This will create A_reparam.sph and A_filtered.sph in the 
working directory.