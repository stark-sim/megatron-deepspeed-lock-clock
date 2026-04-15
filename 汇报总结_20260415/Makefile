# Makefile for Chinese presentation

# Config
LATEX = xelatex
MAIN = main_zh
OUTPUT = GPU能效优化_20%+功耗节省方案

# Default target
all: $(OUTPUT).pdf

# Build presentation with XeLaTeX for Chinese support
$(OUTPUT).pdf: $(MAIN).tex
	@echo "Building Chinese presentation with XeLaTeX..."
	$(LATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(LATEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	mv $(MAIN).pdf $(OUTPUT).pdf
	@echo "✓ Generated: $(OUTPUT).pdf"

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.fdb_latexmk *.fls

# Clean everything including PDF
distclean: clean
	rm -f $(OUTPUT).pdf

# Quick compile (single pass)
quick:
	$(LATEX) $(MAIN).tex

# View PDF (macOS)
view: $(OUTPUT).pdf
	open "$(OUTPUT).pdf"

# Count slides
slides:
	@grep -c "begin{frame}" $(MAIN).tex | xargs echo "Total slides:"

# Check for issues
lint:
	@echo "Checking for common issues..."
	@grep -n "TODO" $(MAIN).tex || echo "No TODOs found"
	@grep -n "FIXME" $(MAIN).tex || echo "No FIXMEs found"

.PHONY: all clean distclean quick view slides lint
