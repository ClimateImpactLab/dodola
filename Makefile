test:
	pytest -vv
format:
	black dodola
format-check:
	flake8 --count --show-source --statistics
	black -v --check dodola
