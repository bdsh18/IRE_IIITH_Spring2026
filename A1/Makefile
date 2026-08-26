.PHONY: data q2-bm25 q3-semantic q4-eval mind-submission ebnerd-submission test
PYTHON ?= ./venv/bin/python

data:
	$(PYTHON) build_pipeline.py --source-root .

q2-bm25:
	$(PYTHON) bm25_retrieval.py --store data/processed

q3-semantic:
	$(PYTHON) semantic_retrieval.py --source-root . --store data/processed

q4-eval:
	$(PYTHON) offline_evaluation.py --store data/processed

mind-submission:
	$(PYTHON) make_mind_submission.py

ebnerd-submission:
	$(PYTHON) make_ebnerd_submission.py
