# Assumptions

TailSafe is a simplified engineering model. This file lists every simplifying
assumption in plain language so users can judge where results may be optimistic
or pessimistic. Numeric values and their sources live in
[`config/params.yaml`](../config/params.yaml); run
`tailsafe params list --assumptions` for the values that still need a citation.

## General

- All occupants are synthetic, generated from the priors in `params.yaml`.
  No personal data is used.
- The model is a decision-support prototype, not a code-compliance tool.

## Building model

- Procedural buildings are *generic* typologies, not surveys of real blocks.
  Layout dimensions (flat width/depth, core size) are template arguments;
  code-driven dimensions (stair width, door widths, riser/going, floor heights,
  refuge interval) come from `params.yaml` and are mostly still assumptions.
- A storey count includes the ground floor.
- Stair travel distance per storey is computed from riser and going: dog-leg
  stairs are two flights plus a semicircular turn on the half landing; scissor
  and straight stairs are one flight. Walking speeds on stairs are measured
  along this line of travel.
- Staircases discharge directly to outside at G/F.
- On refuge floors staircases continue through; occupants can step out into the
  refuge area but are not forced to transfer between stairs (the HK requirement
  should be checked and the template updated if needed).
- The `twin_core` typology is interpreted as a tower whose central core holds a
  *twin pair of interlocking (scissor) staircases*, each reached through a
  protected lobby, above a podium with no residents. If "twin-core" was meant as
  two separate cores, a second core can be added to the template.
- Every lift stops at every level; one lift per building is the firefighting lift.
- Doors are always openable; their fire rating and self-closing flags are
  recorded for the hazard model (M4).
