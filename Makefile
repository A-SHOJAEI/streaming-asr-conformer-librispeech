PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
.PHONY: setup data train eval report all clean
.NOTPARALLEL:

CONFIG ?= configs/smoke.yaml
PYTHON := .venv/bin/python

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	bash scripts/bootstrap_venv.sh

data: setup
	$(PYTHON) -m streaming_asr.cli.data --config $(CONFIG)

train: setup
	$(PYTHON) -m streaming_asr.cli.train --config $(CONFIG)

eval: setup
	$(PYTHON) -m streaming_asr.cli.eval --config $(CONFIG)

report: setup
	$(PYTHON) -m streaming_asr.cli.report --config $(CONFIG)

all: data train eval report

clean:
	rm -rf runs artifacts/*.json artifacts/*.md
