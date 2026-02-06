#!/bin/bash
# Run Pyserini LMDirichletSimilarity baseline on fast evaluation set (12 smaller datasets)

export EVAL_EXCLUDE_DATASETS="dl19,dl20,fever,climate-fever,hotpotqa,dbpedia-entity,nq,quora,webis-touche2020,cqadupstack,leetcode,aops,theoremqa_questions,robotics,psychology,sustainable_living"

# Run Pyserini baseline
uv run python evaluator_ql_parallel.py pyserini \
  --save results/baselines/ql_pyserini.json \
  --verbose
