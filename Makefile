TESTDATA=tests/input

dev:
	pip3 install -r requirements-dev.txt
	pip3 install -e .
	pre-commit install

dev_extra:
	pip3 install -r requirements-dev.txt
	pip3 install -e .[spacy_ja]
	pip3 install -e .[spacy_zh]
	pre-commit install

test: dev check docs
	pip3 install -e .
	pytest tests

check:
	isort -rc -c src/
	isort -rc -c tests/
	black src/ --check
	black tests/ --check
	flake8 src/
	flake8 tests/
	mypy src/

docs:
	sphinx-build -W -b html docs/ _build/html

clean:
	pip3 uninstall -y fonduer
	rm -rf src/fonduer.egg-info
	rm -rf _build/

.PHONY: dev dev_extra test clean check docs
