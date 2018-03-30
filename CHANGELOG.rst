.. note::
    Fonduer is still under active development and APIs may still change
    rapidly.

Version 0.1.5 
-------------
.. warning::
    This release is NOT backwards compatable with v0.1.4. Specifically, in order
    to initialize a session with postgresql, you no longer do

    .. code:: python
        
        os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + DBNAME
        from fonduer import SnorkelSession       
        session = SnorkelSession()

    which had the side-effects of manipulating your database tables on import.
    Now, you use the Meta class to initialize your session:

    .. code:: python

        from fonduer import Meta       
        session = Meta.init("postgres://localhost:5432/" + DBNAME).SnorkelSession()
      
    No side-effects occur until ``Meta`` is initialized.

* `@lukehsiao`_: Remove reliance on environment vars and remove side-effects of
  importing fonduer (`#36 <https://github.com/HazyResearch/fonduer/pull/36>`_)

Version 0.1.4 
-------------

* `@lukehsiao`_: Separate tutorials into their own repo (`#31
  <https://github.com/HazyResearch/fonduer/pull/31>`_)

Version 0.1.3
-------------

Minor hotfix to the README formatting for PyPi.

Version 0.1.2
-------------

* `@lukehsiao`_: Deploy Fonduer to PyPi using Travis-CI 


.. 
  For convenience, all username links for contributors can be listed here

.. _@lukehsiao: https://github.com/lukehsiao
