#!/usr/bin/env bash

# source setup.sh
export DIR_PWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH}:$DIR_PWD"
else
  export PYTHONPATH="$DIR_PWD"
fi

echo "${PYTHONPATH}"
