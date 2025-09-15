SHELL:=/bin/bash
EXPORT_REQ_NAME:=exported-requirements.txt
# Search for the project name and the version in the pyproject.toml file
PROJECT_NAME?=$(shell awk '/^\[project\]/ {flag=1; next} /^\[/ {flag=0} flag && /^name = / {print}' pyproject.toml | sed -n 's/^name = "\([^"]*\)"/\1/p')
VERSION?=$(shell grep -oP '(?<=version = ")[0-9.]+' pyproject.toml)

# Set cache directories
CACHE_DIR:=.cache
PARTIAL_CACHE_PATHS:="fl-manager,huggingface"

# Project dependencies
GROUP_DEPS_DEV:=$(shell grep '^GROUP_DEPS_DEV=' config/dependencies.env | cut -d '=' -f2 | tr -d '\"' | sed 's/[^ ]\+/\-\-group &/g')
EXTRA_DEPS_DEV?=$(shell grep '^EXTRA_DEPS_DEV=' config/dependencies.env | cut -d '=' -f2 | tr -d '\"' | sed 's/[^ ]\+/\-\-extra &/g')
GROUP_DEPS_PRO:=$(shell grep '^GROUP_DEPS_PRO=' config/dependencies.env | cut -d '=' -f2 | tr -d '\"' | sed 's/[^ ]\+/\-\-group &/g')
EXTRA_DEPS_PRO?=$(shell grep '^EXTRA_DEPS_PRO=' config/dependencies.env | cut -d '=' -f2 | tr -d '\"' | sed 's/[^ ]\+/\-\-extra &/g')

# permissions for the user in the container
HOST_USER?=$(shell whoami)
U_ID?=$(shell id -u)
G_ID?=$(shell id -g)
IS_ROOTLESS:=$(shell docker info -f "{{println .SecurityOptions}}" | grep rootless > /dev/null 2>&1 && echo 1 || echo 0)
GPU_ARGS:=$(shell nvidia-smi > /dev/null 2>&1 && echo "--gpus all" || echo "")

.PHONY: \
	default \
	_build _run \
	dev/build dev/run dev/provision dev/docs dev/jupyter \
	dev/test dev/test-cov \
	dev/liccheck \
	pro/build pro/run pro/check \
	run/sonar \
	package_manager/run package_manager/export_reqs \
	whl/build \
	load-pre-commit update-pre-commit format lint \
	executable_scripts \
	setup_uv_cache_dir setup_uv_share_dir \
	fl_clean clean help

default: help

##@
##@ Available commands
##@
_build:
	DOCKER_BUILDKIT=1 docker build \
		$(BUILD_KWARGS) \
		--build-arg STAGE=$(STAGE) \
		--build-arg GROUP_DEPENDENCIES="$(GROUP_DEPENDENCIES)" \
		--build-arg EXTRA_DEPENDENCIES="$(EXTRA_DEPENDENCIES)" \
		-t $(PROJECT_NAME):$(STAGE) . \
		-f devops/containers/Dockerfile

_run: executable_scripts
	docker run --rm $(shell [ -n "$(CMD)" ] && echo "-t" || echo "-it") \
		$(GPU_ARGS) \
		--init \
		--network host \
		-e IS_ROOTLESS=$(IS_ROOTLESS) \
		-e HOST_USER=$(HOST_USER) -e U_ID=$(U_ID) -e G_ID=$(G_ID) \
		-e HOME=/home/$(HOST_USER) \
		-v $(shell pwd):/opt/project \
		--entrypoint ./bin/runtime_host_user_entrypoint.sh \
		$(shell ./bin/gen_cache_dir_mounts.sh -u ${HOST_USER} -b $(CACHE_DIR) -d "${PARTIAL_CACHE_PATHS}" -c) \
		--name $(PROJECT_NAME)-$(STAGE)$(shell [ -n "$(NAME)" ] && echo "-$(NAME)" || echo "")-container \
		$(PROJECT_NAME):$(STAGE) $(or $(CMD), bash)

dev/build: STAGE = dev
dev/build: GROUP_DEPENDENCIES = $(GROUP_DEPS_DEV)
dev/build: EXTRA_DEPENDENCIES = $(EXTRA_DEPS_DEV)
dev/build: _build ##@ Build dev image

dev/run: STAGE = dev
dev/run: dev/build _run ##@ Run interactive docker session with dev image and source code mounted

dev/provision: NAME = provision
dev/provision: CMD = "cd ./examples/${EXAMPLE} && nvflare provision"
dev/provision: dev/run ##@ Run interactive docker session with dev image and source code mounted

dev/compose:
	@(cd ./examples/${EXAMPLE} && docker compose ${CMD})

dev/compose-up: CMD = "up"
dev/compose-up: dev/compose ##@ Run federated environment

dev/docs: NAME = mkdocs
dev/docs: CMD = mkdocs serve -f docs/mkdocs.yml
dev/docs: dev/run ##@ Run mkdocs server

dev/jupyter: NAME = jupyter
dev/jupyter: PORT = $(shell [ -n "$(JUPYTER_PORT)" ] && echo "$(JUPYTER_PORT)" || echo "8888")
dev/jupyter: CMD = jupyter lab --port=$(PORT) --no-browser --allow-root --notebook-dir='notebooks'
dev/jupyter: dev/run ##@ Run jupyter server

dev/test: NAME = pytest
dev/test: CMD = pytest -ra packages/
dev/test: dev/run ##@ Run tests inside docker

dev/test-cov: NAME = pytest-cov
dev/test-cov: CMD = pytest --cov-report=xml:reports/coverage.xml --junitxml=reports/unit-tests.xml --cov=fl_manager -ra --disable-warnings packages/
dev/test-cov: dev/run ##@ Run tests inside docker

dev/liccheck: NAME = license-check
dev/liccheck: CMD = liccheck --no-deps -r $(EXPORT_REQ_NAME); rm $(EXPORT_REQ_NAME)
dev/liccheck: package_manager/export_reqs dev/run ##@ Run license checks inside docker

pro/build: STAGE = pro
pro/build: GROUP_DEPENDENCIES = $(GROUP_DEPS_PRO)
pro/build: EXTRA_DEPENDENCIES = $(EXTRA_DEPS_PRO)
pro/build: _build  ##@ Build prod image

pro/run: STAGE = pro
pro/run: pro/build _run ##@ Run interactive docker session with pro image

AWK_CHECK_HELPER_CMD = awk '{\
	hash = substr($$0, 1, 64);\
	fullpath = substr($$0, 67);\
	split(fullpath, parts, "/");\
	f_name = parts[length(parts)];\
	print hash, f_name\
}'
pro/check: ##@ Validate project code sha256 inside pro image and actual code
	@echo	dev - $(shell docker run --rm -v $(shell pwd):/opt/project --entrypoint bash fl-manager:dev -c '\
		find /opt -path "*/fl_manager/*" -type f -name "*.py" -not -path "*/tests/*" -exec sha256sum {} +'\
		| $(AWK_CHECK_HELPER_CMD) | sort -k2 | sha256sum | cut -d ' ' -f1)
	@echo	pro - $(shell docker run --rm --entrypoint bash fl-manager:pro -c '\
		find /opt -path "*/fl_manager/*" -type f -name "*.py" -exec sha256sum {} +'\
		| $(AWK_CHECK_HELPER_CMD) | sort -k2 | sha256sum | cut -d ' ' -f1)

run/sonar: ##@ Run SonarQube Analysis
	@(sed -i 's|<source>/opt/project/fl_manager</source>|<source>/usr/src/fl_manager</source>|g' reports/coverage.xml && \
	cp sonar-project.properties sonar-project.properties.backup && \
	grep '^#sonar.' .env | sed 's/#//g' >> sonar-project.properties && \
	trap 'echo Caught SIGINT; mv sonar-project.properties.backup sonar-project.properties' INT && \
	docker run --rm --network host -e SONAR_HOST_URL="http://localhost:9000" -v $(shell pwd):/usr/src sonarsource/sonar-scanner-cli; \
	mv sonar-project.properties.backup sonar-project.properties)

##@
##@ Packaging management
##@
package_manager/run: setup_uv_cache_dir setup_uv_share_dir ##@ Run interactive docker session with uv image and source code mounted
	docker run --rm -it \
		$(GPU_ARGS) \
		-e IS_ROOTLESS=$(IS_ROOTLESS) \
		-e HOST_USER=$(HOST_USER) -e U_ID=$(U_ID) -e G_ID=$(G_ID) \
		-e HOME=$(HOME) \
		-v $(HOME)/.cache/uv:$(HOME)/.cache/uv \
		-v $(HOME)/.local/share/uv:$(HOME)/.local/share/uv \
		-v $(shell pwd):$(HOME)/$(PROJECT_NAME) \
		--env-file ./config/uv.env \
		-e UV_PROJECT_ENVIRONMENT=$(HOME)/.venv \
		--entrypoint $(HOME)/$(PROJECT_NAME)/bin/runtime_host_user_entrypoint.sh \
		--name $(PROJECT_NAME)-package-manager-container \
		$(shell if $(ENV_FILE),--env-file $(ENV_FILE),) \
		ghcr.io/astral-sh/uv:bookworm "cd ${HOME}/${PROJECT_NAME} && $(or $(CMD), bash)"

package_manager/export_reqs: CMD = uv export --format requirements.txt --no-editable --no-hashes --no-dev --all-packages --no-emit-workspace -o $(EXPORT_REQ_NAME)
package_manager/export_reqs: package_manager/run  ##@ Run uv to export requirements

##@
##@ Building commands
##@
whl/build: CMD = uv build --all-packages --wheel
whl/build: package_manager/run  ##@ Run uv to build wheel packages of the project

##@
##@ Pre-commit commands
##@
load-pre-commit: ##@ Install pre-commit hooks
	@(pre-commit install && pre-commit install --hook-type commit-msg)

update-pre-commit: load-pre-commit ##@ Update pre-commit hooks
	@(pre-commit autoupdate)

format: load-pre-commit	##@ Run code auto-formatters
	@(git ls-files | xargs pre-commit run ruff-format --files)

lint: load-pre-commit ##@ Run linters: pre-commit
	@(git ls-files | xargs pre-commit run --show-diff-on-failure --files)

##@
##@ Misc commands
##@
executable_scripts: ##@ Make scripts executable
	@(chmod 755 bin/*.sh)

setup_uv_cache_dir: ##@ Create, if doesn't exist, '~/.cache/uv' directory
	@([ -d ~/.cache/uv ] || mkdir -p ~/.cache/uv)

setup_uv_share_dir: ##@ Create, if doesn't exist, '~/.local/share/uv' directory
	@([ -d ~/.local/share/uv ] || mkdir -p ~/.local/share/uv)

fl_clean: ##@ Remove FL workspace
	@(docker run --rm -v $(shell pwd):/app bash -c "find /app/examples -type d -name workspace -prune -exec rm -rf {} \;")

clean: ##@ Remove artifacts
	@(rm -rf .Trash-$(U_ID))
	@(rm -rf .pytest_cache && rm -rf .ruff_cache)
	@(rm -rf reports && rm -f .coverage)
	@(rm -rf dist)

# Adapted from: https://gist.github.com/prwhite/8168133?permalink_comment_id=4718682#gistcomment-4718682
help: ##@ (Default) Print listing of key targets with their descriptions
	@printf "\nUsage: make <command>\n"
	$(eval FORMAT_HELP_SPACES=$(shell grep -h "##@" Makefile | sed -e 's/:.*//' | grep -v "##@" | awk '{ print length }' | sort -n | tail -1))
	@grep -F -h "##@" $(MAKEFILE_LIST) | grep -F -v grep -F | sed -e 's/\\$$//' | sed -e 's/:.*##@/ ##@/' | awk -v spacing="$(FORMAT_HELP_SPACES)" 'BEGIN {FS = ":*[[:space:]]*##@[[:space:]]*"}; \
	{ \
		if($$2 == "") \
			pass; \
		else if($$0 ~ /^#/) \
			printf "\n%s\n", $$2; \
		else if($$1 == "") \
			printf "     %-*s%s\n", "", spacing + 4, $$2; \
		else \
			printf "    \033[34m%-*s\033[0m %s\n", spacing + 4, $$1, $$2; \
	}'
