# AWG Segment Factory

A library that allows you to programatically define AWG segments in a nice way.

## To run

This packages uses uv to manage its dependencies. To run any of the examples, one can call:
```
uv run examples/recreate_current.py
```

### Restricted environments (optional)

If your environment blocks access to `~/.cache` (e.g. some sandboxes/CI), run uv with a repo-local cache:

```
uv run --cache-dir .uv-cache examples/recreate_current.py
```

uv will be managing a virtualenvironment sitting at in `.venv` and can be updated through `uv sync`. You probably want the dev dependencies (related to jupyter/matplotlib debugging) when working with this package. These will automatically be installed with `uv sync`.
