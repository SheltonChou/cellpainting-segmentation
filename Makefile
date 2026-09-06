.PHONY: verify check-data figures statistics

verify:
	python scripts/verify_results.py

check-data:
	python scripts/check_data.py

statistics:
	python scripts/statistical_tests.py

figures:
	python scripts/paper_result_figures.py
	python scripts/final_visualisation.py
