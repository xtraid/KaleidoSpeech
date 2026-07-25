.PHONY: init test redis redis-check vector-sync api producer build-demo \
	evaluate-demo frontend frontend-bg dev dev-reload

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
	uv run --env-file .env uvicorn app.api:app --host 127.0.0.1 --port 8000

producer:
	uv run python -m app.audio_producer

build-demo:
	uv run python -m scripts.build_acoustic_benchmarks \
		--limit-per-word 100 tmp/1.zip tmp/2.zip tmp/3.zip

evaluate-demo:
	@echo "Usage: uv run python -m scripts.demo_evaluate WORD FILE.wav"
	@echo "Worker equivalent: uv run python -m app.worker evaluate-wav WORD FILE.wav"

frontend:
	cd frontend && python3 -m http.server 8765

frontend-bg:
	cd frontend && python3 -m http.server 8765 >/tmp/advx-frontend.log 2>&1 &
	@echo "Frontend running at http://localhost:8765"

dev:
	@echo "Starting API on :8000 and frontend on :8765"
	@trap 'kill $$api_pid 2>/dev/null || true' EXIT INT TERM; \
		uv run --env-file .env uvicorn app.api:app --host 127.0.0.1 --port 8000 & \
		api_pid=$$!; \
		cd frontend && python3 -m http.server 8765

dev-reload:
	uv run --env-file .env uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
