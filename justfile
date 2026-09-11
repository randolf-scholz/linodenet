#!/usr/bin/env just --justfile
mod docs
mod tests

export GIT_ROOT := `git rev-parse --show-toplevel`
export GIT_PREFIX := `git rev-parse --show-prefix`
export TEST_DIR := GIT_ROOT / "tests"

[default]
[doc('List available commands.')]
help:
    @just --justfile {{ justfile() }} --list --list-submodules --unsorted

[doc('Remove generated Python cache and build artifacts.')]
clean:
    uv run pyclean {{ GIT_ROOT }} --debris

[doc('Build release source and wheel distributions.')]
build-release:
    uv build -Ccmake.build-type=Release

[doc('Build release source and wheel distributions.')]
build: build-release

[doc('Build a debug wheel.')]
build-debug:
    uv build --wheel -Ccmake.build-type=Debug

[doc('Install the project in editable mode.')]
install:
    uv pip install --editable .

[doc('Clean the workspace, create a uv virtualenv, and install the project editable.')]
setup:
    #!/usr/bin/env bash
    set -eu

    git clean -fdx
    uv venv
    . .venv/bin/activate
    uv pip install --editable .
    git submodule update --init --recursive

[doc('Configure git remotes for this repository.')]
setup-remote:
    #!/usr/bin/env bash
    set -eu

    #HILDESHEIM="https://software.ismll.uni-hildesheim.de/ISMLL-internal/time-series/linodenet.git"
    GITHUB="https://github.com/randolf-scholz/linodenet.git"

    echo -e "\nCurrent remotes:"
    git remote -v

    echo -e "\nDeleting all remotes..."
    for remote_name in $(git remote); do
        git remote remove "${remote_name}"
    done

    echo -e "\nAdding remote ${GITHUB}..."
    git remote add origin "$GITHUB"
    git remote set-url --add --push origin "$GITHUB"
    #git remote set-url --add --push origin "$HILDESHEIM"

    #echo -e "\nAdding remote ${HILDESHEIM}..."
    #git remote add hildesheim "$HILDESHEIM"
    #git remote set-url --add --push hildesheim "$HILDESHEIM"
    #git remote set-url --add --push hildesheim "$GITHUB"

    echo -e "\nSetting default remote:"
    git fetch origin
    git branch --set-upstream-to=origin/main main
    git push -u origin --all

    echo -e "\nNew remote config:"
    git remote -v
