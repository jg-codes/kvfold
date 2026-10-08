table1:
	python scripts/reproduce_table1.py

test:
	python -m pytest -q

.PHONY: table1 test
