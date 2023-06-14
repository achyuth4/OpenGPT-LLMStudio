PYTHON_VERSION ?= 3.10
PYTHON ?= python$(PYTHON_VERSION)
PIP ?= $(PYTHON) -m pip
PIPENV ?= $(PYTHON) -m pipenv
PIPENV_PYTHON = $(PIPENV) run python
PIPENV_PIP = $(PIPENV_PYTHON) -m pip
PWD = $(shell pwd)

PHONY: pipenv
pipenv:
	$(PIP) install pip --upgrade
	$(PIP) install pipenv==2022.10.4

.PHONY: setup
setup: pipenv
	$(PIPENV) install --verbose --python $(PYTHON_VERSION)
	$(PIPENV_PIP) install https://github.com/h2oai/wave/releases/download/nightly/h2o_wave-nightly-py3-none-manylinux1_x86_64.whl --force-reinstall

.PHONY: setup-dev
setup-dev: pipenv
	$(PIPENV) install --verbose --dev --python $(PYTHON_VERSION)
	$(PIPENV_PIP) install https://github.com/h2oai/wave/releases/download/nightly/h2o_wave-nightly-py3-none-manylinux1_x86_64.whl --force-reinstall

.PHONY: export-requirements
export-requirements: pipenv
	$(PIPENV) requirements > requirements.txt
	 echo "https://github.com/h2oai/wave/releases/download/nightly/h2o_wave-nightly-py3-none-manylinux1_x86_64.whl" >> requirements.txt

clean-env:
	$(PIPENV) --rm

clean-data:
	rm -rf data

clean-output:
	rm -rf output

reports:
	mkdir -p reports

.PHONY: style
style: reports pipenv
	@echo -n > reports/flake8.log
	@echo -n > reports/mypy.log
	@echo

	-$(PIPENV) run flake8 | tee -a reports/flake8.log
	@echo

	-$(PIPENV) run mypy . --check-untyped-defs | tee -a reports/mypy.log
	@echo

	@if [ -s reports/flake8.log ]; then exit 1; fi

.PHONY: format
format: pipenv
	$(PIPENV) run isort .
	$(PIPENV) run black .

.PHONY: isort
isort: pipenv
	$(PIPENV) run isort .

.PHONY: black
black: pipenv
	$(PIPENV) run black .

.PHONY: test
test: reports
	export PYTHONPATH=$(PWD) && $(PIPENV) run pytest -v -s -x \
		--junitxml=./reports/junit.xml \
		tests/* | tee reports/pytest.log

.PHONY: wave
wave:
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=True \
	H2O_WAVE_PRIVATE_DIR="/download/@$(PWD)/output/download" \
	$(PIPENV) run wave run app

.PHONY: wave-no-reload
wave-no-reload:
	H2O_WAVE_MAX_REQUEST_SIZE=25MB \
	H2O_WAVE_NO_LOG=True \
	H2O_WAVE_PRIVATE_DIR="/download/@$(PWD)/output/download" \
	$(PIPENV) run wave run --no-reload app

.PHONY: shell
shell:
	$(PIPENV) shell

setup-doc:  # Install documentation dependencies
	cd documentation && npm install 

run-doc:  # Run the doc locally
	cd documentation && npm start

update-documentation-infrastructure:
	cd documentation && npm update @h2oai/makersaurus
	cd documentation && npm ls

build-doc-locally:  # Bundles your website into static files for production 
	cd documentation && npm run build

serve-doc-locally:  # Serves the built website locally 
	cd documentation && npm run serve
