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

## Population and behaviour

- Households move together at the pace of their slowest member; nobody splits
  from their household.
- Ages and wheelchair use are drawn independently per household member (a
  household made only of children gets an adult). The 65+ share knob applies to
  residents; the share among people *present* differs by time of day.
- A live-in domestic helper is present per the helper probabilities and the
  time-slot presence rates. Helpers are excluded from the 65+ census share.
- Wheelchair users above ground never walk down alone. They are carried by
  their household if an able adult or helper is present (with a probability), or
  wait for a lift (only when lifts are in evacuation service) or for rescue.
  Frail older adults walk, slowly, unless lifts are available.
- A household's fatigue follows the member who will be slowest over the full
  descent from their floor.
- Care-home staff each escort one room; staff do not make repeated trips.

## Mesoscopic simulator

- Walkers can overtake each other freely on an arc. On narrow stairs this is
  optimistic; slow walkers raise the walking density and so slow others, but
  they do not block them.
- Density slows walkers only up to the flow-maximising density; beyond that,
  congestion shows up as queues and spillback.
- Everyone knows the fastest route to an exit (by estimated travel time) and
  learns about blockages only on reaching them.
- Doors are always open to flow; closing and hold-open policies act only through
  the hazard model (M4).
- The fire service rescues households one at a time, lowest floor first, using
  the stairs without interfering with the crowd.
- Lifts board households first come, first served and carry them to the
  discharge level without intermediate pick-ups.

## Scenarios and risk metrics

- In the baseline, lifts are **not** used for evacuation (usual Hong Kong
  practice; lifts are homed for fire-service use). "Lifts out of service"
  therefore only matters when a spec puts lifts into evacuation service.
- A named stair blockage makes every flight of that stair impassable from its
  time onward (e.g. "smoke-logged"). Occupants already on it keep going.
- The fire floor is sampled and recorded but only affects outcomes once the
  hazard model (M4) is enabled.
- Fire-service rescue starts at a time drawn from
  `rescue.operations_start`; rescue parameters are assumptions and dominate the
  `total_time` tail. Read `self_evacuation_time` alongside it.
- Non-finite losses (nobody should be left inside without hazards) would be
  counted as censored and capped, making tail statistics lower bounds.

## Hazard model (simplified — not a substitute for CFD)

- One fire, in a flat, growing as t² to a constant peak; no decay, no flashover
  dynamics, no suppression by sprinklers or firefighters.
- Each space is one well-mixed zone: no hot upper layer, so smoke is spread
  over the whole height of a room or corridor (optimistic early, pessimistic
  later in tall spaces).
- Transport uses fixed exchange flows with assumed velocities; wind, outside
  temperature and pressurisation systems are not modelled explicitly (the stack
  bias is a crude stand-in). HK's warm climate may give weaker stack effects.
- Self-closing doors are shut except for a constant "in use" fraction for
  stair doors; the fire flat's door is either open or shut for the whole fire.
- Species scale with one products tracer (soot, CO, CO₂ yields fixed); O₂
  depletion, HCN, irritants and radiant heat are ignored.
- Occupants do not turn back from smoke-logged stairs by themselves; stair
  loss is modelled with explicit blockages (e.g. the demo's Stair A at 4 min).
- The fire flat's household reacts quickly (short pre-movement); everyone else
  hears the alarm at ignition.
- Numbers are only as good as the assumed parameters in `params.yaml`
  (`hazard.*`). Import CFD results through `tailsafe.hazard.external` when
  available.

## Interventions

- **Evacuation lifts** carry only households with a mobility-impaired member
  (wheelchair users, frail older adults) who choose to wait for them
  (`behaviour.use_evacuation_lift_probability`). The firefighting lift is never
  used, lifts are not affected by smoke, and they start after a fixed
  switch-over delay. With the "wheelchair users only" rule, households with a
  frail older adult walk (slowly) instead of waiting. Letting frail residents
  wait for a lift can shorten the total evacuation time yet lengthen the time
  for 95% of occupants to get out, when too few lifts serve too many floors.
- **Floor wardens** are trained residents or staff already on their floor at
  the alarm. They knock on every door of their floor and the floors next to
  it within a fixed sweep time, and escort one household that cannot use the
  stairs alone. Wardens themselves are not simulated as extra occupants.
- **Phased release** holds whole floor bands for a delay; held residents wait
  in their flats (and still accumulate smoke dose there).
- **Stair assignment** is followed by everyone on the assigned floors; if the
  assigned stair is blocked they fall back to the fastest route.
- **Holding stair doors open** restores their full flow capacity and lets smoke
  pass freely (both through the zone model).

## Microscopic replay engine

- People are discs of radius `movement.micro.body_radius` × √(space factor);
  households do not hold together once they have left their flat (they
  start together and walk at the household's pace).
- Rooms are the bounding rectangles of node polygons; walls are the
  rectangle sides; there is no furniture. Buildings without room polygons
  (graph-only cases) cannot be replayed.
- A stair flight is a straight strip of lanes; turning at half-landings is
  folded into the flight length (as in the meso engine). Up and down
  movement use separate lanes, so counter-flow on stairs does not collide.
- Doorway capacity comes from lanes and headways, not from a specific-flow
  table; the fundamental diagram and the agreement study in
  `docs/validation.md` show how this compares with the hydraulic model.
- Lift trips, fire-service rescues and toxic dose come from the meso run of
  the same scenario.
- All `movement.micro.*` values are assumptions (no calibration yet).
