TESTDATA=tests/input

dev:
	pip install -r requirements-dev.txt
	pip install -e .

test: dev check
	pytest tests -v -rsXx	

check: dev 
	isort -rc -c fonduer/
	black fonduer/ --check

clean:
	pip uninstall fonduer
	rm -r fonduer.egg-info

.PHONY: dev test clean check
