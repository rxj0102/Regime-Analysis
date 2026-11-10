.PHONY: reproduce render check test

reproduce:  ## execute the notebook from data/snapshot/ and refresh the README
	cd notebooks && uv run --with nbconvert --with ipykernel jupyter nbconvert --to notebook --execute regime_analysis.ipynb --output /tmp/regime_executed.ipynb --ExecutePreprocessor.timeout=1200
	uv run python scripts/render_readme.py

render:
	uv run python scripts/render_readme.py

check:
	uv run python scripts/render_readme.py --check

test:
	uv run python -m pytest -q
