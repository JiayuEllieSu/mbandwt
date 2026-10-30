# Contributing to mbandwt

Bug reports and changes with accompanying tests are welcome.

## Local setup

```bash
python -m pip install -e ".[test,plot]"
python -m pytest -q
```

## Development guidelines

- Preserve the filtering/decimation/delay conventions described in
  [`docs/conventions.md`](docs/conventions.md). Update documentation and
  tests when a convention changes.
- Include a regression test for any bug fix or new bank family.
- Validate new filter coefficients and note their source; do not copy
  unsourced published coefficient tables.
- Keep the core dependency set limited to NumPy and SciPy.
- Check the polyphase perfect-reconstruction condition in addition to
  round-trip tests.

## Scope

The package currently covers finite-signal M-band transforms and
perfect-reconstruction FIR filter banks.
