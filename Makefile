.PHONY: setup relevance train rank all

setup:
	python -m pip install -r requirements.txt

relevance:
	python relevance/src/calculate_rrf_relevance.py

train:
	python src/train_technical_depth.py

rank:
	python src/calculate_critic_ranking.py

all:
	python src/run_pipeline.py
