test:
	pytest -vv
format:
	black dodola
format-check:
	flake8 --count --show-source --statistics dodola
	black -v --check dodola
docs:
	mkdocs build
