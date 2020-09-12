dev:
	pip install -r requirements-dev.txt
	pip install -e . --use-feature=2020-resolver
	pre-commit install

dev_extra:
	pip install -r requirements-dev.txt
	pip install -e .[spacy_ja,spacy_zh] --use-feature=2020-resolver
	pre-commit install

test: dev check docs
	pytest tests

check:
	isort -c src/
	isort -c tests/
	black src/ --check
	black tests/ --check
	flake8 src/
	flake8 tests/
	mypy src/

docs:
	sphinx-build -W -b html docs/ _build/html

clean:
	pip uninstall -y fonduer
	rm -rf src/fonduer.egg-info
	rm -rf _build/

.PHONY: dev dev_extra test clean check docs
