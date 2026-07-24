.PHONY: init test redis redis-check vector-sync api producer build-demo evaluate-demo

init:
	uv sync
	uv run python -m scripts.init_db

test:
	uv run pytest

redis:
	docker compose up -d redis

redis-check:
	uv run python -m scripts.check_redis

vector-sync:
	uv run python -m scripts.sync_vector_index

api:
	uv run uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload

producer:
	uv run python -m app.audio_producer

build-demo:
	uv run python -m scripts.build_acoustic_benchmarks \
		--limit-per-word 100 tmp/1.zip tmp/2.zip tmp/3.zip

evaluate-demo:
	@echo "Usage: uv run python -m scripts.demo_evaluate WORD FILE.wav"
	@echo "Worker equivalent: uv run python -m app.worker evaluate-wav WORD FILE.wav"
