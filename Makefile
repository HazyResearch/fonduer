TESTDATA=tests/input

dev: 
	pip install -e .

test: dev check
	pytest tests -v -rsXx	

check:
	flake8 fonduer --count --max-line-length=127 --statistics --ignore=E731,W503,E741,E123,E203

clean:
	pip uninstall fonduer
	rm -r fonduer.egg-info

.PHONY: dev test clean check
