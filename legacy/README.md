# Legacy (frozen)

Frozen reference only. Do not edit it, do not import from it.

Known defect: the trained model has 8 output classes while app.py assumes 2, so it raises IndexError on some inputs. See docs/legacy-findings.md.
