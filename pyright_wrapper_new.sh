#!/bin/env bash

# check that the git root is relative to the current path PWD
# run git root, terminate early if it fails
git_root=$(git rev-parse --show-toplevel) || exit 1
if [[ ! "$PWD" =~ ^"$git_root".* ]]; then
    echo "Error: git root is not a parent of the current path";
    exit 1;
fi

# ANSI color codes with escaping '['.
red="\x1b\[31m"
yellow="\x1b\[33m"
blue="\x1b\[34m"
teal="\x1b\[36m"
default="\x1b\[39m"
grey="\x1b\[90m"

# ANSI color codes without escaping '['.
BLUE_BOLD="\x1b[1;34m"
RED_BOLD="\x1b[1;31m"
TEAL_BOLD="\x1b[1;36m"
RESET="\x1b[0m"  # resets all attributes

# line + column position.
position_regex="${yellow}\d+${default}:${yellow}\d+${default}"
# capture path without the git root.
path_group="${git_root}/(?P<path>[^\r\n]*?:${position_regex})"

fail_pattern="${red}error${default}"
warn_pattern="${teal}warning${default}"
info_pattern="${blue}information${default}"
ecode_pattern="${grey}\x20*\(report\w+\)${default}";

# capture only the first line of the error message and the error code
fail_group="(?P<fail>${fail_pattern}: [^\r\n]*?)(?:[\r\n].*?)?(?P<fail_code>${ecode_pattern})"
warn_group="(?P<warn>${warn_pattern}: [^\r\n]*?)(?:[\r\n].*?)?(?P<warn_code>${ecode_pattern})"
info_group="(?P<info>${info_pattern}: [^\r\n]*)"

# final regex and output format
regex="\x20+${path_group} - (?:${fail_group}|${warn_group}|${info_group})";
output="\$path - \$fail\$fail_code\$warn\$warn_code\$info";

# handling of arguments (needed to deal with files with spaces)
cmd="$(printf '%q ' pyright "$@")";

# run the command and capture the exit code
result="$(
  script /dev/null -efqc "$cmd" |  # -e: forward exit code, -q: quiet, -f: flush output
  rg --multiline --multiline-dotall "$regex" -or "$output";
  exit "${PIPESTATUS[0]}"  # capture the exit code of pyright
)";
pyright_exit_code=$?;
echo "$result";

# summary statistics
fail_count=$(echo -n "$result" | rg -c --include-zero "${fail_pattern}");
warn_count=$(echo -n "$result" | rg -c --include-zero "${warn_pattern}");
info_count=$(echo -n "$result" | rg -c --include-zero "${info_pattern}");
echo -e -n "${RED_BOLD}${fail_count} error${RESET}, "
echo -e -n "${TEAL_BOLD}${warn_count} warning${RESET}, "
echo -e "${BLUE_BOLD}${info_count} information${RESET}"

# check that pyright_exit_code agrees with (fail_count == 0):
if [ "$pyright_exit_code" -ne $(( fail_count > 0 )) ]; then
    echo "Error: pyright exit code does not agree with error count!";
    echo "pyright_exit_code: $pyright_exit_code, fail_count: $fail_count";
    echo "Please file a bug report.";
    exit 1;
fi

exit "$pyright_exit_code"
