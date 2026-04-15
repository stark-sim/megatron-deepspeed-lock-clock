# Makefile for presentation

# Config
LATEX = pdflatex
MAIN = main_en
OUTPUT = GPU_Energy_Optimization_20Percent_Savings

# Default target
all: $(OUTPUT).pdf

# Build presentation
$(OUTPUT).pdf: $(MAIN).tex
	@echo "Building presentation..."
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

# View PDF (macOS)
view: $(OUTPUT).pdf
	open $(OUTPUT).pdf

# Count slides
slides:
	@grep -c "begin{frame}" $(MAIN).tex | xargs echo "Total slides:"

# Check for issues
lint:
	@echo "Checking for common issues..."
	@grep -n "TODO" $(MAIN).tex || echo "No TODOs found"
	@grep -n "FIXME" $(MAIN).tex || echo "No FIXMEs found"

.PHONY: all clean distclean view slides lint
