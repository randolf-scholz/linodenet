[doc('Sphinx documentation commands.')]
mod docs

[default]
[doc('Show available recipes.')]
help:
    @just --justfile {{justfile()}} --list

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
