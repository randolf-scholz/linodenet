# Minimal makefile for Sphinx documentation

export PYTORCH_JIT=0

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -w .sphinx/warnings.txt
SPHINXBUILD   ?= sphinx-build
DOCDIR        = .
BUILDDIR      = ./_build
SOURCEDUR     = ../src
TESTDIR       = ../tests
VENVDIR       = ../.venv
VENV          = $(VENVDIR)/bin/activate


.PHONY: help clean apidoc build-html Makefile


# Put it first so that "make" without argument is like "make help".
help: $(VENVDIR)
	@. $(VENV) && $(SPHINXBUILD) -M help "$(DOCDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	# backup previous build
	mkdir -p "$(BUILDDIR)"
	rm -rf _previous_build
	mv "$(BUILDDIR)" _previous_build

	# remove previous build
	rm -rf "$(BUILDDIR)"
	rm -rf _autosummary
	rm -rf autoapi
	rm -rf apidoc
	rm -rf api

build-html:
	@$(SPHINXBUILD) -M html "$(DOCDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

apidoc:
	@$(MAKE) clean
	@. $(VENV) && sphinx-apidoc  --separate  -o apidoc/  ../src/linodenet
	@$(MAKE) build-html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(MAKE) clean
	@. $(VENV) && $(SPHINXBUILD) -M $@ "$(DOCDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
