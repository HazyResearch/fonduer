TESTDATA=tests/input

dev: docs
		pip install -e .

test: docs 
	  pytest tests -v -rsXx	

docs:
		pandoc --from=markdown --to=rst --output=README.rst README.md

clean:
		rm -f README.rst

.PHONY: dev test docs clean
