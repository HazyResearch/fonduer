|Fonduer|
=========

|CI-CD| |CodeClimate| |Codecov| |ReadTheDocs| |PyPI| |PyPIVersion| |GitHubStars| |License| |CodeStyle|

Fonduer is a Python package and framework for building knowledge base
construction (KBC) applications from **richly formatted data**.

Note that Fonduer is still *actively under development*, so feedback and
contributions are welcome. Submit bugs in the Issues_ section or feel free to
submit your contributions as a pull request.

Getting Started
---------------

Check out our `Getting Started Guide`_ to get up and running with Fonduer.

Learning how to use Fonduer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Fonduer tutorials`_ cover the Fonduer workflow, showing how to extract
relations from hardware datasheets and scientific literature.

Reference
---------

`Fonduer: Knowledge Base Construction from Richly Formatted
Data <https://arxiv.org/abs/1703.05028>`__ (`blog <https://hazyresearch.stanford.edu/fonduer>`__)::

    @inproceedings{wu2018fonduer,
      title={Fonduer: Knowledge Base Construction from Richly Formatted Data},
      author={Wu, Sen and Hsiao, Luke and Cheng, Xiao and Hancock, Braden and Rekatsinas, Theodoros and Levis, Philip and R{\'e}, Christopher},
      booktitle={Proceedings of the 2018 International Conference on Management of Data},
      pages={1301--1316},
      year={2018},
      organization={ACM}
    }


Acknowledgements
----------------

Fonduer leverages the work of Emmental_ and Snorkel_.


.. |CodeClimate| image:: https://img.shields.io/codeclimate/maintainability/HazyResearch/fonduer.svg
   :alt: Code Climate
   :target: https://codeclimate.com/github/HazyResearch/fonduer
.. |Fonduer| image:: docs/static/img/fonduer-logo.png
   :target: https://github.com/HazyResearch/fonduer
.. |CI-CD| image:: https://img.shields.io/github/workflow/status/HazyResearch/fonduer/ci.svg
   :target: https://github.com/HazyResearch/fonduer/actions
.. |Codecov| image:: https://img.shields.io/codecov/c/github/HazyResearch/fonduer
   :target: https://codecov.io/gh/HazyResearch/fonduer
.. |ReadTheDocs| image:: https://img.shields.io/readthedocs/fonduer.svg
   :target: https://fonduer.readthedocs.io/
.. |PyPI| image:: https://img.shields.io/pypi/v/fonduer.svg
   :target: https://pypi.org/project/fonduer/
.. |PyPIVersion| image:: https://img.shields.io/pypi/pyversions/fonduer.svg
   :target: https://pypi.org/project/fonduer/
.. |GitHubStars| image:: https://img.shields.io/github/stars/HazyResearch/fonduer.svg
   :target: https://github.com/HazyResearch/fonduer/stargazers
.. |License| image:: https://img.shields.io/github/license/HazyResearch/fonduer.svg
   :target: https://github.com/HazyResearch/fonduer/blob/master/LICENSE
.. |CodeStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black

.. _Emmental: https://github.com/senwu/emmental/
.. _Snorkel: https://hazyresearch.github.io/snorkel/
.. _Issues: https://github.com/HazyResearch/fonduer/issues/
.. _Getting Started Guide: https://fonduer.readthedocs.io/en/latest/user/getting_started.html
.. _Fonduer tutorials: https://github.com/hazyresearch/fonduer-tutorials
.. _Mailing List: https://groups.google.com/forum/#!forum/fonduer-dev
