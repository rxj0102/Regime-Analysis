.PHONY: reproduce render check test

reproduce:  ## execute the notebook on open data, write results/ and outputs/, refresh the README
	MPLBACKEND=Agg uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute --inplace 07_regime_analysis.ipynb --ExecutePreprocessor.timeout=1200
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	uv run python -m pytest -q
