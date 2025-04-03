# build is docker compose build
all:
	docker compose run --user `id -u` cmems bash -c "ruff format . && pytest && ruff check . && mypy ."

test:
	docker compose run --user `id -u` cmems python3 -m pytest -s

test-watch:
	docker compose run --user `id -u` cmems ptw

lint:
	docker compose run --user `id -u` cmems bash -c "ruff format . && ruff check . && mypy ."

lint-watch:
	docker compose run --user `id -u` cmems bash -c "watch -n1 bash -c \"'ruff format . && ruff check . && mypy .'\""

upgrade-packages:
	docker compose run --user 0 cmems bash -c "python3 -m pip install pip-upgrader && pip-upgrade --skip-package-installation"

bash:
	docker compose run --user `id -u` cmems bash

up:
	docker compose up

build:
	docker compose build
