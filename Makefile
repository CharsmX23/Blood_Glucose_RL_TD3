.PHONY: setup test pid smoke train compare clean all

setup:
	uv sync --extra dev

test:
	uv run pytest -v

pid:
	uv run python -m eval.compare --pid-only

smoke:
	uv run python -m train.train_td3 --seed 0 --timesteps 20000
	uv run python -m eval.compare --seeds 0

train:
	uv run python -m train.train_all --seeds 0 1 2 --timesteps 300000

compare:
	uv run python -m eval.compare --seeds 0 1 2

tb:
	uv run tensorboard --logdir results/

clean:
	rm -rf results/*_seed* results/figures
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: test pid
