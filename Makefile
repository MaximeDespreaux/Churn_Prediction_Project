# Shortcuts for the churn prediction project (GNU make).
#
#   make                                   list the targets (same as `make help`)
#   make <target> VARIABLE=value           override a variable, e.g.
#   make docker-build DOCKER_USERNAME=me   -> docker build --target app --tag me/churn-prediction-app:latest .
#   make docker-pipeline CMD=evaluate      -> docker compose run --rm pipeline evaluate

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

UV              ?= uv
DOCKER_USERNAME ?= local
IMAGE           ?= $(DOCKER_USERNAME)/churn-prediction-app
PIPELINE_IMAGE  ?= $(DOCKER_USERNAME)/churn-prediction-pipeline
TEST_IMAGE      ?= $(DOCKER_USERNAME)/churn-prediction-test
TAG             ?= latest
CONTAINER       ?= churn-app
PORT            ?= 8501
CMD             ?= --help

# compose.yaml names its images ${DOCKER_USERNAME:-local}/churn-prediction-<service>:${TAG:-latest}
export DOCKER_USERNAME TAG

.PHONY: help install install-app lock hooks lint format test test-fast app \
    check features evaluate train predict tune \
    docker-build docker-build-pipeline docker-build-test docker-run docker-smoke docker-stop \
    docker-push docker-push-pipeline \
    compose-up compose-watch compose-down docker-build-all docker-test docker-test-fast \
    docker-lint docker-pipeline \
    clean

##@ General

help: ## Show this help
	@printf 'Usage: make <target> [VARIABLE=value]\n'
	@awk 'BEGIN { FS = ":.*## " } /^##@ / { printf "\n%s\n", substr($$0, 5) } /^[a-zA-Z0-9_-]+:.*## / { printf "  %-22s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf '\nVariables: UV=%s TAG=%s CONTAINER=%s PORT=%s CMD=%s\n' '$(UV)' '$(TAG)' '$(CONTAINER)' '$(PORT)' '$(CMD)'
	@printf '           IMAGE=%s PIPELINE_IMAGE=%s TEST_IMAGE=%s\n' '$(IMAGE)' '$(PIPELINE_IMAGE)' '$(TEST_IMAGE)'

##@ Setup

install: ## Install app, model libraries, Optuna and dev tools (uv sync --all-extras)
	$(UV) sync --all-extras

install-app: ## Install only the app and data pipeline dependencies (no extras, no dev tools)
	$(UV) sync --no-dev

lock: ## Update uv.lock after editing pyproject.toml
	$(UV) lock

hooks: ## Install the pre-commit hooks (checks run on every commit)
	$(UV) run pre-commit install

##@ Quality

lint: ## Check linting and formatting (ruff, black)
	$(UV) run ruff check .
	$(UV) run black --check .

format: ## Fix lint issues and format the code (ruff --fix, black)
	$(UV) run ruff check --fix .
	$(UV) run black .

check: ## Run every pre-commit hook on all files (what CI runs)
	$(UV) run pre-commit run --all-files --show-diff-on-failure

test: ## Run the test suite with coverage (fails below 100 %; report + coverage.xml)
	$(UV) run pytest --cov --cov-report=term-missing --cov-report=xml

test-fast: ## Run the tests that do not fit models (-m "not model")
	$(UV) run pytest -m "not model"

##@ App

app: ## Run the Streamlit app on http://localhost:8501
	$(UV) run streamlit run app/streamlit_app.py

##@ Pipeline (`churn` CLI; needs `make install`; evaluate, train and tune fit models)

features: ## Build the feature tables: data/raw/ -> data/processed/
	$(UV) run churn build-features

# evaluate, train and tune fit models: they are slow and need the ml/tune extras.
evaluate: ## Hold-out evaluation -> reports/metrics.json (fits models)
	$(UV) run churn evaluate

train: ## Fit the ensemble -> models/churn_models.joblib (fits models)
	$(UV) run churn train

predict: ## Score the test users -> reports/submissions/submission_ensemble.csv (after train)
	$(UV) run churn predict --submission reports/submissions/submission_ensemble.csv

tune: ## Optuna search, 50 trials -> models/params/*.json (fits models)
	$(UV) run churn tune --n-trials 50 --write

##@ Docker images

docker-build: ## Build the app image IMAGE:TAG
	docker build --target app --tag $(IMAGE):$(TAG) .

docker-build-pipeline: ## Build the pipeline image PIPELINE_IMAGE:TAG
	docker build --target pipeline --tag $(PIPELINE_IMAGE):$(TAG) .

docker-build-test: ## Build the test image TEST_IMAGE:TAG
	docker build --target test --tag $(TEST_IMAGE):$(TAG) .

docker-run: ## Run IMAGE:TAG in the background as CONTAINER on http://localhost:PORT
	docker run --detach --name $(CONTAINER) --publish $(PORT):8501 $(IMAGE):$(TAG)

docker-smoke: ## Wait up to 60 s for the running app to be healthy (prints logs on failure)
	@for i in $$(seq 1 30); do \
		if curl --silent --fail http://localhost:$(PORT)/_stcore/health; then \
			echo " -> app is healthy on http://localhost:$(PORT)"; \
			exit 0; \
		fi; \
		sleep 2; \
	done; \
	echo "App not healthy after 60 s; logs of $(CONTAINER):"; \
	docker logs $(CONTAINER) || true; \
	exit 1

docker-stop: ## Stop and remove CONTAINER
	docker rm --force $(CONTAINER)

docker-push: ## Push IMAGE:TAG (run `docker login` first)
	docker push $(IMAGE):$(TAG)

docker-push-pipeline: ## Push PIPELINE_IMAGE:TAG (run `docker login` first)
	docker push $(PIPELINE_IMAGE):$(TAG)

##@ Docker Compose (services: app, pipeline, tests)

compose-up: ## Build and start the app with Docker Compose
	docker compose up --build --detach

compose-watch: ## Run the app and sync/rebuild it on code changes
	docker compose watch

compose-down: ## Stop the Compose app and remove its volumes
	docker compose down --volumes

docker-build-all: ## Build the app, pipeline and test images
	docker compose --profile pipeline --profile test build

docker-test: ## Run the test suite in the test image
	docker compose run --rm tests

docker-test-fast: ## Run the tests that do not fit models in the test image
	docker compose run --rm tests pytest -m "not model"

docker-lint: ## Check linting and formatting in the test image (ruff, black)
	docker compose run --rm tests sh -c "ruff check . && black --check ."

docker-pipeline: ## Run a `churn` command in the pipeline image, e.g. make docker-pipeline CMD=evaluate
	docker compose run --rm pipeline $(CMD)

##@ Housekeeping

clean: ## Remove caches, coverage and build files (never data/ or models/)
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	find src app tests -type d -name __pycache__ -prune -exec rm -rf {} +
	find src -maxdepth 1 -type d -name '*.egg-info' -prune -exec rm -rf {} +
