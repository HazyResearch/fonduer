TESTDATA=tests/input

dev: docs
		pip install -e .

test: docs dev check
	  pytest tests -v -rsXx	

check:
	  flake8 fonduer --count --max-line-length=127 --statistics --ignore=E731,W503,E741,E123

docs:
		pandoc --from=markdown --to=rst --output=README.rst README.md

clean:
		rm -f README.rst

.PHONY: dev test docs clean check
