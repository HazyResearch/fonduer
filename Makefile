TESTDATA=tests/input

dev: 
	pip install -e .

test: dev check
	pytest tests -v -rsXx	

check:
	flake8 fonduer --count --max-line-length=127 --statistics --ignore=E731,W503,E741,E123

clean:
	rm fonduer.egg-info
	pip uninstall fonduer

.PHONY: dev test clean check
