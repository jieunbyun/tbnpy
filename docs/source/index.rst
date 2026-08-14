.. tbnpy documentation master file, created by
   sphinx-quickstart on Mon Jan  5 15:51:12 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tbnpy documentation
===================

**TBNpy** is a Python toolkit for tensor-based Bayesian network (TBN).
TBN tensorlises the quantification and inferences of Bayesian networks (BNs) and therby addresses BNs' scalability issues when applied to **large-scale systems** (i.e. high-dimensional system events).

Background
----------

**Bayesian network** is a probabilistic graphical model that can map the causality between variables, for example:

.. figure:: _static/bn_illust.png
   :width: 300
   :align: center

   An illustration of a Bayesian network.

For further reading:

* For a brief introduction of Bayesian network in the context of system reliability, refer to `this blog article <https://jieunbyun.github.io/categories/bn/>`_.
* A formal publication is under preparation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


   getting_started/overview
   getting_started/repo_structure
   api/modules
   examples/EMA_aftershock
   examples/abcde_example

