.PHONY: help paper figures test lint verify benchmark clean

help:
	@echo "paper      compile the manuscript"
	@echo "figures    regenerate standalone manuscript figures"
	@echo "test       run the simulation test suite"
	@echo "lint       check simulation lint and formatting"
	@echo "verify     run lint and tests"
	@echo "benchmark  run a bounded simulation performance benchmark"
	@echo "clean      remove LaTeX and Python build caches"

paper:
	cd manuscript && latexmk -pdf main.tex

figures:
	python simulation/scripts/generate_figure_1_response_rules.py
	python simulation/scripts/generate_figure_2_nutrient_landscape.py
	python simulation/scripts/generate_figure_4_macroscopic_observables.py --reuse
	python simulation/scripts/generate_figure_5_foraging_observables.py --reuse
	python simulation/scripts/generate_figure_6_information_observables.py --reuse

test:
	$(MAKE) -C simulation test

lint:
	$(MAKE) -C simulation lint

verify:
	$(MAKE) -C simulation verify

benchmark:
	$(MAKE) -C simulation benchmark

clean:
	cd manuscript && latexmk -c
	$(MAKE) -C simulation clean
