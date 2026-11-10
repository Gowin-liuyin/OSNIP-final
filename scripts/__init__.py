# Makes `scripts` a regular package so the repo's scripts win over any
# unrelated top-level `scripts` package installed in site-packages
# (regular packages shadow namespace packages regardless of sys.path order).
