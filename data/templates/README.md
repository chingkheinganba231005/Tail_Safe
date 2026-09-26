# Building templates

Example buildings produced by the procedural Hong Kong typology generators in
`tailsafe/building/templates.py`. Files here are validated by the test suite
against `schemas/building.schema.json`.

Regenerate or create variants with the CLI, e.g.:

```bash
.venv/bin/tailsafe building templates                       # list templates + options
.venv/bin/tailsafe building generate cruciform --storeys 40 --out data/templates/cruciform_40.json
.venv/bin/tailsafe building generate twin_core --set podium_levels=4 --out out/tower.json
.venv/bin/tailsafe building render data/templates/care_home_4.json --out out/care_home.png
```

Only the small care-home example is committed; larger buildings are cheap to
regenerate (well under a second) and deterministic.
