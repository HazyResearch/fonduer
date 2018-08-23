TESTDATA=tests/input

dev:
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test: dev check docs
	pytest tests

check:
	isort -rc -c fonduer/
	isort -rc -c tests/
	black fonduer/ --check
	black tests/ --check
	flake8 fonduer/
	flake8 tests/

docs:
	sphinx-build -W -b html docs/ _build/html

clean:
	pip uninstall -y fonduer
	rm -rf fonduer.egg-info
	rm -rf _build/

.PHONY: dev test clean check docs
