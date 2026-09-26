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
