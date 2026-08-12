# HSTU and Linear Attention

This directory contains the technical note comparing HSTU's pointwise SiLU
attention with feature-map and delta-rule linear attention.

Build it from this directory with the cluster's TinyTeX installation:

```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
latexmk -pdf -interaction=nonstopmode main_arxiv.tex
```

The generated document is `main_arxiv.pdf`.
