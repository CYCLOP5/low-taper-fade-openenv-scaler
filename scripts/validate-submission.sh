#!/usr/bin/env bash

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BOLD=''
  NC=''
fi

run_with_timeout() {
  local secs="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() {
  rm -f "${CLEANUP_FILES[@]+${CLEANUP_FILES[@]}}"
}
trap cleanup EXIT

log() {
  printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"
}

pass() {
  log "${GREEN}PASSED${NC} -- $1"
}

fail() {
  log "${RED}FAILED${NC} -- $1"
}

hint() {
  printf "  ${YELLOW}Hint:${NC} %b\n" "$1"
}

stop_at() {
  printf "\n${RED}${BOLD}Validation stopped at %s.${NC}\n" "$1"
  exit 1
}

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your Space runtime URL such as https://your-space.hf.space\n"
  printf "  repo_dir   Path to your repo default current directory\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/4: Pinging live Space health${NC}"

HEALTH_OUTPUT="$(portable_mktemp validate-health)"
CLEANUP_FILES+=("$HEALTH_OUTPUT")
HEALTH_CODE="$(curl -s -o "$HEALTH_OUTPUT" -w "%{http_code}" "$PING_URL/health" --max-time 30 2>/dev/null || printf "000")"

if [ "$HEALTH_CODE" = "200" ]; then
  pass "HF Space responds to /health"
else
  fail "HF Space /health returned HTTP $HEALTH_CODE"
  hint "Use the runtime URL ending in .hf.space not the huggingface.co/spaces page URL"
  stop_at "Step 1"
fi

log "${BOLD}Step 2/4: Pinging live Space reset${NC}"

RESET_OUTPUT="$(portable_mktemp validate-reset)"
CLEANUP_FILES+=("$RESET_OUTPUT")
RESET_CODE="$(curl -s -o "$RESET_OUTPUT" -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 2>/dev/null || printf "000")"

if [ "$RESET_CODE" = "200" ]; then
  pass "HF Space responds to /reset"
else
  fail "HF Space /reset returned HTTP $RESET_CODE"
  hint "Check the Space logs for sandbox or filesystem setup failures"
  hint "If /health works but /reset fails the issue is likely runtime sandbox setup not model API credentials"
  stop_at "Step 2"
fi

log "${BOLD}Step 3/4: Running docker build${NC}"

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker or run this step on a machine with Docker available"
  stop_at "Step 3"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/"
  stop_at "Step 3"
fi

BUILD_OUTPUT="$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1)"
BUILD_OK=$?

if [ "$BUILD_OK" -eq 0 ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed timeout=${DOCKER_BUILD_TIMEOUT}s"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 3"
fi

log "${BOLD}Step 4/4: Running openenv validate${NC}"

if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  hint "Install it with pip install openenv-core"
  stop_at "Step 4"
fi

VALIDATE_OUTPUT="$(cd "$REPO_DIR" && openenv validate 2>&1)"
VALIDATE_OK=$?

if [ "$VALIDATE_OK" -eq 0 ]; then
  pass "openenv validate succeeded"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 4"
fi

printf "\n${GREEN}${BOLD}All submission checks passed.${NC}\n"
