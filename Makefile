.PHONY: help install dev-install test test-verbose test-coverage clean build publish format lint type-check all

# 기본 명령어 (make 만 실행 시)
.DEFAULT_GOAL := help

# Python 인터프리터
PYTHON := python3
PIP := $(PYTHON) -m pip

# 프로젝트 정보
PACKAGE_NAME := nadoo-flow-core
SRC_DIR := src/nadoo_flow
TEST_DIR := tests
EXAMPLES_DIR := examples

##@ 도움말

help:  ## 사용 가능한 명령어 표시
	@echo "Nadoo Flow Core - Build & Package Commands"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ 설치

install:  ## 패키지 설치 (production)
	$(PIP) install -e .

dev-install:  ## 개발 의존성 포함 설치
	$(PIP) install -e ".[dev]"

##@ 테스트

test:  ## 테스트 실행
	pytest $(TEST_DIR) -v

test-verbose:  ## 상세 출력과 함께 테스트 실행
	pytest $(TEST_DIR) -vv -s

test-coverage:  ## 커버리지 리포트와 함께 테스트
	pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-watch:  ## 파일 변경 시 자동 테스트 (pytest-watch 필요)
	ptw $(TEST_DIR) -- -v

##@ 코드 품질

format:  ## 코드 포맷팅 (black + isort)
	black $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	isort $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

lint:  ## 린팅 (ruff + flake8)
	ruff check $(SRC_DIR) $(TEST_DIR)
	flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=120

type-check:  ## 타입 체크 (mypy)
	mypy $(SRC_DIR) --strict

check-all: format lint type-check test  ## 모든 품질 체크 실행

##@ 빌드

clean:  ## 빌드 아티팩트 제거
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(TEST_DIR)/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf .coverage

build: clean  ## 배포 패키지 빌드
	$(PYTHON) -m build

##@ 배포

publish-test:  ## TestPyPI에 배포
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish:  ## PyPI에 배포 (주의!)
	@echo "  Warning: Publishing to PyPI. Press Ctrl+C to cancel..."
	@sleep 3
	$(PYTHON) -m twine upload dist/*

##@ 개발

run-examples:  ## 모든 예제 실행
	@echo "Running examples..."
	@for example in $(EXAMPLES_DIR)/*.py; do \
		echo "\n=== Running $$example ===" ; \
		$(PYTHON) $$example || exit 1 ; \
	done

docs:  ## 문서 생성 (MkDocs - 권장)
	mkdocs build

docs-serve:  ## 문서 로컬 서버 실행 (MkDocs)
	mkdocs serve

docs-sphinx:  ## 문서 생성 (Sphinx)
	cd docs && make html

docs-sphinx-serve:  ## 문서 로컬 서버 실행 (Sphinx)
	cd docs/build/html && $(PYTHON) -m http.server 8000

docs-deploy:  ## 문서 GitHub Pages 배포 (MkDocs)
	mkdocs gh-deploy --force

##@ 전체 워크플로우

all: clean install dev-install format lint type-check test build  ## 전체 워크플로우 실행

ci:  ## CI 환경에서 실행할 체크
	@echo "Running CI checks..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test-coverage
	@echo "✅ All CI checks passed!"

##@ 유틸리티

version:  ## 현재 버전 표시
	@$(PYTHON) -c "from nadoo_flow import __version__; print(__version__)"

deps-update:  ## 의존성 업데이트
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev]"

tree:  ## 프로젝트 구조 트리 출력
	@tree -I '__pycache__|*.pyc|*.egg-info|.git|.pytest_cache|.mypy_cache|.ruff_cache|htmlcov' -L 3
