@echo off
set PYTHONPATH=.
python scripts/run_recommendation.py --config config.yaml --data data/processed/recommendation_example1.csv --mode train --output output/
pause
