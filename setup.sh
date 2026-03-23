#!/usr/bin/env bash

# source setup.sh
export DIR_PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH}:$DIR_PWD"
else
  export PYTHONPATH="$DIR_PWD"
fi

# Avoid HuggingFace tokenizers fork/parallelism warning in DataLoader workers.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "${PYTHONPATH}"
