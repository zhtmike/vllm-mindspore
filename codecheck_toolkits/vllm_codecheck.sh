pip install -r requirements-lint.txt

RET_FLAG=0

cd ..
# yapf formats code automatically

MERGEBASE="$(git merge-base origin/master HEAD)"
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  yapf --in-place --recursive --parallel --exclude build/
fi

if [[ $? -ne 0 ]]; then
  echo "yapf run failed."
  RET_FLAG=1
else
  echo "yapf run success."
fi

# codespell check
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  codespell --skip ./vllm_mindspore/ops/ascendc/*
fi
if [[ $? -ne 0 ]]; then
  echo "codespell check failed."
  RET_FLAG=1
else
  echo "codespell check success."
fi

# ruff check
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  ruff check
fi
if [[ $? -ne 0 ]]; then
  echo "ruff check failed."
  RET_FLAG=1
else
  echo "ruff check success."
fi

# isort fixed
if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  isort
fi
if [[ $? -ne 0 ]]; then
  echo "isort fixed failed."
  RET_FLAG=1
else
  echo "isort fixed success."
fi

# mypy check type

PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &> /dev/null; then
  git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
  mypy --follow-imports skip --python-version "${PYTHON_VERSION}" "$@"
fi
if [[ $? -ne 0 ]]; then
  echo "mypy check failed."
  RET_FLAG=1
else
  echo "mypy check success."
fi

cd - || exit $RET_FLAG
