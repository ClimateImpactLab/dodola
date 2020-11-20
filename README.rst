dodola
======

Prototype application for GCM bias-correction and downscaling

This is an unstable prototype. This is under heavy development.

Features
--------

* Nothing! The unit tests might work if you're lucky.

Example
-------

After installing, use from the commandline with::

    dodola biascorrect <inputURL> <modeltrainigURL> <obstrainingURL> <outputURL>

See more help with::

    dodola --help    

Installation
------------

You shouldn't! This will likely run within a Docker container in a production environment on cloud infrastructure. But, to install with ``pip``::

    pip install git+https://github.com/ClimateImpactLab/dodola
