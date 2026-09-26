"""Numba kernel of the mesoscopic queue-network simulator.

The model is described in :mod:`tailsafe.sim.meso`; this module only holds the
compiled time-stepping loop. Everything is plain arrays so it runs in Numba's
nopython mode (set ``NUMBA_DISABLE_JIT=1`` to step through it in Python).

Entities are *groups* (households moving together). A group is always in
exactly one place:

* ``PREMOVE`` / ``DWELL`` — at a node, waiting until ``g_ready``;
* ``POOL`` — at a node, ready to leave (FIFO list per node);
* ``ON_ARC`` — walking along an arc (position ``g_pos``);
* ``QUEUED`` — at the end of an arc, waiting to enter the next one (FIFO per arc);
* ``WAIT_LIFT`` / ``IN_LIFT`` — lift service;
* ``WAIT_RESCUE`` — waiting for the fire service;
* ``EXITED`` / ``INCAPACITATED`` — terminal.

Lists are singly linked through ``g_next``; entries whose state no longer
matches the list (rescued or incapacitated groups) are skipped lazily.
"""

from __future__ import annotations

import numpy as np
from numba import njit

ST_PREMOVE = 0
ST_ON_ARC = 1
ST_QUEUED = 2
ST_POOL = 3
ST_DWELL = 4
ST_WAIT_LIFT = 5
ST_IN_LIFT = 6
ST_WAIT_RESCUE = 7
ST_EXITED = 8
ST_INCAPACITATED = 9

LP_IDLE = 0
LP_TO_PICKUP = 1
LP_TO_DISCHARGE = 2
LP_OUT = 3

LIFT_TOP_DOWN = 0
LIFT_NEAREST = 1
LIFT_BOTTOM_UP = 2

MODE_WAIT_RESCUE = 3

# Special results of _desired_arc
_NO_ROUTE = -1
_STRANDED = -3

META_T_END = 0
META_STEPS = 1
META_N_REC = 2
META_FORCED = 3
META_LIFT_TRIPS = 4
META_N_RESCUED = 5
META_SIZE = 8


@njit(cache=True)
def _push(head, tail, nxt, key, g):  # type: ignore[no-untyped-def]
    nxt[g] = -1
    if tail[key] < 0:
        head[key] = g
    else:
        nxt[tail[key]] = g
    tail[key] = g


@njit(cache=True)
def _pop(head, tail, nxt, key):  # type: ignore[no-untyped-def]
    g = head[key]
    if g >= 0:
        head[key] = nxt[g]
        if head[key] < 0:
            tail[key] = -1
        nxt[g] = -1
    return g


@njit(cache=True)
def _speed_factor(density, a, d0, floor):  # type: ignore[no-untyped-def]
    """Hydraulic speed-density factor, with density capped at the flow-maximising
    value 1/(2a): denser crowds are represented by queues, not by slower walking."""
    if density <= d0:
        return 1.0
    d_crit = 0.5 / a
    if density > d_crit:
        density = d_crit
    f = (1.0 - a * density) / (1.0 - a * d0)
    return f if f > floor else floor


@njit(cache=True)
def _desired_arc(  # type: ignore[no-untyped-def]
    g,
    v,
    t,
    state,
    g_know,
    g_class,
    g_leg,
    g_wp,
    g_wp_kind,
    wp_target,
    wp_next,
    next_arc,
    arc_blocked_from,
):
    """Arc group ``g`` wants to take from node ``v`` (or -1 / -3)."""
    while g_leg[g] < g_wp.shape[1] and g_wp[g, g_leg[g]] >= 0:
        w = g_wp[g, g_leg[g]]
        b = wp_next[w, v]
        if wp_target[w] != v and b >= 0 and arc_blocked_from[b] > t:
            return b
        if g_wp_kind[g, g_leg[g]] == 1:
            return _STRANDED  # cannot reach the lift lobby: wait for rescue here
        g_leg[g] += 1
    k = g_know[g]
    c = g_class[g]
    b = next_arc[k, c, v]
    if b < 0:
        b = next_arc[k, 0, v]
    if b >= 0 and arc_blocked_from[b] <= t:
        g_know[g] = state  # discover every blockage known so far
        b = next_arc[state, c, v]
        if b < 0:
            b = next_arc[state, 0, v]
        if b >= 0 and arc_blocked_from[b] <= t:
            b = _NO_ROUTE
    return b


@njit(cache=True)
def run_kernel(  # type: ignore[no-untyped-def]
    # ---- network
    arc_src,
    arc_dst,
    arc_kind,
    arc_len,
    arc_area,
    arc_store,
    arc_cap,
    arc_rev,
    arc_mw,
    arc_blocked_from,
    in_ptr,
    in_arc,
    node_is_exit,
    node_level,
    node_pool_w,
    # ---- routing
    next_arc,
    state_time,
    wp_target,
    wp_next,
    # ---- groups
    g_start,
    g_ready0,
    g_space,
    g_count,
    g_vh,
    g_vdown,
    g_vup,
    g_fmin,
    g_fefold,
    g_class,
    g_mode,
    g_home_level,
    g_wp,
    g_wp_dwell,
    g_wp_kind,
    # ---- lifts
    lift_stop_node,
    lift_discharge_node,
    lift_discharge_level,
    lift_cap,
    lift_speed,
    lift_from,
    lift_until,
    level_elev,
    lift_door_time,
    lift_board_time,
    lift_overhead,
    lift_rule,
    # ---- rescue
    rescue_start,
    n_teams,
    rescue_climb,
    rescue_carry,
    rescue_handling,
    # ---- hazard
    hz_dt,
    hz_speed,
    hz_fed,
    fed_threshold,
    # ---- movement parameters
    dens_a,
    dens_d0,
    speed_floor,
    cf_penalty,
    # ---- numerics
    dt,
    t_max,
    rec_every,
    stuck_time,
    # ---- outputs
    out_exit,
    out_left,
    out_fed,
    out_rescued,
    out_state,
    out_arc_maxq,
    out_arc_qint,
    out_arc_entries,
    out_series,
    out_meta,
):
    """Advance the evacuation until every group is done or ``t_max`` is reached."""
    M = arc_src.size
    N = node_is_exit.size
    G = g_start.size
    L = lift_cap.size
    K = state_time.size
    H = hz_speed.shape[0]
    n_rec_max = out_series.shape[1]

    # ---------------------------------------------------------------- state
    g_state = np.empty(G, np.int8)
    g_node = g_start.copy()
    g_arc = np.full(G, -1, np.int32)
    g_pos = np.zeros(G)
    g_clock = np.zeros(G)
    g_arrive = np.zeros(G)
    g_ready = g_ready0.copy()
    g_next = np.full(G, -1, np.int32)
    g_leg = np.zeros(G, np.int32)
    g_know = np.zeros(G, np.int32)
    g_floors = np.zeros(G)
    g_lift = np.full(G, -1, np.int32)

    a_occ = np.zeros(M)
    a_qspace = np.zeros(M)
    a_qpers = np.zeros(M)
    a_budget = arc_cap * dt
    a_credit = np.zeros(M)
    a_qhead = np.full(M, -1, np.int32)
    a_qtail = np.full(M, -1, np.int32)

    n_phead = np.full(N, -1, np.int32)
    n_ptail = np.full(N, -1, np.int32)
    n_credit = np.zeros(N)
    n_demand = np.zeros(N, np.int32)
    w_head = np.full(N, -1, np.int32)
    w_tail = np.full(N, -1, np.int32)
    w_space = np.zeros(N)
    w_claim = np.zeros(N)

    l_phase = np.zeros(L, np.int32)
    l_level = lift_discharge_level.copy()
    l_free = np.zeros(L)
    l_target = np.full(L, -1, np.int32)
    l_claim = np.zeros(L)
    l_load = np.zeros(L)
    l_head = np.full(L, -1, np.int32)
    l_tail = np.full(L, -1, np.int32)
    team_free = np.zeros(max(n_teams, 0))

    arr_g = np.empty(G, np.int32)
    arr_t = np.empty(G)

    n_done = 0
    n_wait_rescue = 0
    forced = 0
    lift_trips = 0
    n_rescued = 0
    for g in range(G):
        out_exit[g] = np.inf
        out_left[g] = -1.0
        out_fed[g] = 0.0
        out_rescued[g] = False
        if g_mode[g] == MODE_WAIT_RESCUE:
            g_state[g] = ST_WAIT_RESCUE
            n_wait_rescue += 1
        else:
            g_state[g] = ST_PREMOVE

    state = 0
    step = 0
    rec = 0
    t = 0.0
    while t < t_max and n_done < G:
        # ------------------------------------------------ 1. network state
        while state + 1 < K and state_time[state + 1] <= t:
            state += 1

        # ------------------------------------------------ 2. ready groups join node pools
        for g in range(G):
            s = g_state[g]
            if (s == ST_PREMOVE or s == ST_DWELL) and g_ready[g] <= t + dt:
                g_state[g] = ST_POOL
                _push(n_phead, n_ptail, g_next, g_node[g], g)
                n_demand[g_node[g]] += 1

        # ------------------------------------------------ 3. replenish inflow capacity
        for a in range(M):
            c = arc_cap[a]
            r = arc_rev[a]
            if r >= 0 and a_occ[r] > 1e-9:
                c *= 1.0 - cf_penalty * a_occ[r] / (a_occ[r] + a_occ[a])
            cd = c * dt
            b = a_budget[a] + cd
            a_budget[a] = b if b < cd else cd

        # ------------------------------------------------ 4. transfers at nodes
        for v in range(N):
            if n_demand[v] <= 0:
                continue
            while True:
                best = -2  # -1 = pool, >= 0 = incoming arc index
                best_credit = -1e300
                total_w = 0.0
                # -- node pool stream
                h = n_phead[v]
                want_pool = -1
                while h >= 0:
                    if g_state[h] != ST_POOL:
                        _pop(n_phead, n_ptail, g_next, v)
                        n_demand[v] -= 1
                        h = n_phead[v]
                        continue
                    b = _desired_arc(
                        h,
                        v,
                        t,
                        state,
                        g_know,
                        g_class,
                        g_leg,
                        g_wp,
                        g_wp_kind,
                        wp_target,
                        wp_next,
                        next_arc,
                        arc_blocked_from,
                    )
                    if b < 0:
                        _pop(n_phead, n_ptail, g_next, v)
                        n_demand[v] -= 1
                        g_state[h] = ST_WAIT_RESCUE
                        n_wait_rescue += 1
                        h = n_phead[v]
                        continue
                    want_pool = b
                    break
                if h >= 0:
                    b = want_pool
                    ok = a_budget[b] > 0.0
                    if ok and a_occ[b] > 0.0 and a_occ[b] + g_space[h] > arc_store[b]:
                        ok = (t - g_ready[h]) >= stuck_time
                    if ok:
                        w = node_pool_w[v]
                        n_credit[v] += w
                        total_w += w
                        if n_credit[v] > best_credit:
                            best_credit = n_credit[v]
                            best = -1
                # -- incoming arc queues
                for idx in range(in_ptr[v], in_ptr[v + 1]):
                    a = in_arc[idx]
                    h = a_qhead[a]
                    while h >= 0 and g_state[h] != ST_QUEUED:
                        _pop(a_qhead, a_qtail, g_next, a)
                        n_demand[v] -= 1
                        h = a_qhead[a]
                    if h < 0:
                        continue
                    b = _desired_arc(
                        h,
                        v,
                        t,
                        state,
                        g_know,
                        g_class,
                        g_leg,
                        g_wp,
                        g_wp_kind,
                        wp_target,
                        wp_next,
                        next_arc,
                        arc_blocked_from,
                    )
                    if b < 0:
                        _pop(a_qhead, a_qtail, g_next, a)
                        n_demand[v] -= 1
                        a_occ[a] -= g_space[h]
                        a_qspace[a] -= g_space[h]
                        a_qpers[a] -= g_count[h]
                        g_state[h] = ST_WAIT_RESCUE
                        g_node[h] = v
                        n_wait_rescue += 1
                        continue
                    ok = a_budget[b] > 0.0
                    if ok and a_occ[b] > 0.0 and a_occ[b] + g_space[h] > arc_store[b]:
                        ok = (t - g_arrive[h]) >= stuck_time
                    if ok:
                        w = arc_mw[a]
                        a_credit[a] += w
                        total_w += w
                        if a_credit[a] > best_credit:
                            best_credit = a_credit[a]
                            best = a
                if best == -2:
                    break
                # -- transfer the head of the chosen stream
                if best == -1:
                    n_credit[v] -= total_w
                    h = _pop(n_phead, n_ptail, g_next, v)
                    t_avail = g_ready[h]
                else:
                    a_credit[best] -= total_w
                    h = _pop(a_qhead, a_qtail, g_next, best)
                    a_occ[best] -= g_space[h]
                    a_qspace[best] -= g_space[h]
                    a_qpers[best] -= g_count[h]
                    t_avail = g_arrive[h]
                n_demand[v] -= 1
                b = _desired_arc(
                    h,
                    v,
                    t,
                    state,
                    g_know,
                    g_class,
                    g_leg,
                    g_wp,
                    g_wp_kind,
                    wp_target,
                    wp_next,
                    next_arc,
                    arc_blocked_from,
                )
                if a_occ[b] > 0.0 and a_occ[b] + g_space[h] > arc_store[b]:
                    forced += 1
                a_budget[b] -= g_space[h]
                a_occ[b] += g_space[h]
                out_arc_entries[b] += g_count[h]
                g_state[h] = ST_ON_ARC
                g_arc[h] = b
                g_pos[h] = 0.0
                g_clock[h] = t_avail if t_avail > t - dt else t
                if out_left[h] < 0.0 and node_level[arc_dst[b]] != g_home_level[h]:
                    out_left[h] = g_clock[h]

        # ------------------------------------------------ 5. lifts
        for li in range(L):
            if l_phase[li] == LP_OUT or t < l_free[li]:
                continue
            active = t >= lift_from[li] and t < lift_until[li]
            if l_phase[li] == LP_TO_DISCHARGE:
                # Arrived at the discharge level: unload.
                persons = 0.0
                h = _pop(l_head, l_tail, g_next, li)
                dnode = lift_discharge_node[li]
                while h >= 0:
                    persons += g_count[h]
                    g_state[h] = ST_DWELL
                    g_node[h] = dnode
                    g_ready[h] = t + lift_board_time * g_count[h]
                    g_lift[h] = -1
                    h = _pop(l_head, l_tail, g_next, li)
                l_level[li] = lift_discharge_level[li]
                l_load[li] = 0.0
                l_free[li] = t + lift_door_time + lift_board_time * persons
                l_phase[li] = LP_IDLE if active else LP_OUT
                continue
            if not active:
                if t >= lift_until[li]:
                    if l_phase[li] == LP_TO_PICKUP:
                        w_claim[lift_stop_node[li, l_target[li]]] -= l_claim[li]
                    l_phase[li] = LP_OUT
                continue
            if l_phase[li] == LP_TO_PICKUP:
                lev = l_target[li]
                node = lift_stop_node[li, lev]
                w_claim[node] -= l_claim[li]
                l_claim[li] = 0.0
                l_level[li] = lev
                persons = 0.0
                while True:
                    h = w_head[node]
                    if h < 0:
                        break
                    if g_state[h] != ST_WAIT_LIFT:
                        _pop(w_head, w_tail, g_next, node)
                        continue
                    if l_load[li] > 0.0 and l_load[li] + g_space[h] > lift_cap[li]:
                        break
                    _pop(w_head, w_tail, g_next, node)
                    w_space[node] -= g_space[h]
                    g_state[h] = ST_IN_LIFT
                    g_lift[h] = li
                    _push(l_head, l_tail, g_next, li, h)
                    l_load[li] += g_space[h]
                    persons += g_count[h]
                    if out_left[h] < 0.0:
                        out_left[h] = t
                if persons > 0.0:
                    lift_trips += 1
                    dz = abs(level_elev[lev] - level_elev[lift_discharge_level[li]])
                    travel = dz / lift_speed[li] + (lift_overhead if dz > 0.0 else 0.0)
                    l_free[li] = t + lift_door_time + lift_board_time * persons + travel
                    l_phase[li] = LP_TO_DISCHARGE
                else:
                    l_phase[li] = LP_IDLE
                    l_free[li] = t
                continue
            # LP_IDLE: choose the next floor to serve.
            n_lv = lift_stop_node.shape[1]
            chosen = -1
            best_key = 1e300
            for lev in range(n_lv):
                node = lift_stop_node[li, lev]
                if node < 0 or lev == lift_discharge_level[li]:
                    continue
                if w_space[node] - w_claim[node] <= 1e-9:
                    continue
                if lift_rule == LIFT_TOP_DOWN:
                    key = -float(lev)
                elif lift_rule == LIFT_BOTTOM_UP:
                    key = float(lev)
                else:
                    key = abs(float(lev - l_level[li]))
                if key < best_key:
                    best_key = key
                    chosen = lev
            if chosen >= 0:
                node = lift_stop_node[li, chosen]
                claim = w_space[node] - w_claim[node]
                if claim > lift_cap[li]:
                    claim = lift_cap[li]
                w_claim[node] += claim
                l_claim[li] = claim
                l_target[li] = chosen
                dz = abs(level_elev[chosen] - level_elev[l_level[li]])
                travel = dz / lift_speed[li] + (lift_overhead if dz > 0.0 else 0.0)
                l_free[li] = t + travel
                l_phase[li] = LP_TO_PICKUP

        # ------------------------------------------------ 6. fire-service rescue
        if n_wait_rescue > 0 and t >= rescue_start:
            any_lift = False
            for li in range(L):
                if l_phase[li] != LP_OUT and lift_until[li] > t:
                    any_lift = True
            for k in range(team_free.size):
                if team_free[k] > t or n_wait_rescue <= 0:
                    continue
                pick = -1
                best_lv = 1 << 30
                for g in range(G):
                    s = g_state[g]
                    if s == ST_WAIT_RESCUE or (s == ST_WAIT_LIFT and not any_lift):
                        lv = node_level[g_node[g]]
                        if lv < best_lv:
                            best_lv = lv
                            pick = g
                if pick < 0:
                    break
                if g_state[pick] == ST_WAIT_LIFT:
                    w_space[g_node[pick]] -= g_space[pick]
                else:
                    n_wait_rescue -= 1
                lv = float(best_lv)
                reach = t + rescue_climb * lv + rescue_handling
                done = reach + rescue_carry * lv
                if out_left[pick] < 0.0:
                    out_left[pick] = reach
                out_exit[pick] = done
                out_rescued[pick] = True
                g_state[pick] = ST_EXITED
                n_done += 1
                n_rescued += 1
                team_free[k] = done

        # ------------------------------------------------ 7. movement
        t_end_step = t + dt
        hi = -1
        if H > 0:
            hi = int(t / hz_dt)
            if hi >= H:
                hi = H - 1
        n_arr = 0
        for g in range(G):
            if g_state[g] != ST_ON_ARC:
                continue
            a = g_arc[g]
            kind = arc_kind[a]
            if kind == 0:
                v0 = g_vh[g]
            elif kind == 1:
                fm = g_fmin[g]
                v0 = g_vdown[g] * (fm + (1.0 - fm) * np.exp(-g_floors[g] / g_fefold[g]))
            else:
                v0 = g_vup[g]
            # Density of people still walking (both directions share the passage).
            # People standing in the end-of-arc queue are congestion already
            # accounted for by capacity, so they don't slow the walkers again.
            occ = a_occ[a] - a_qspace[a]
            r = arc_rev[a]
            if r >= 0:
                occ += a_occ[r] - a_qspace[r]
            spd = v0 * _speed_factor(occ / arc_area[a], dens_a, dens_d0, speed_floor)
            if hi >= 0:
                m1 = hz_speed[hi, arc_src[a]]
                m2 = hz_speed[hi, arc_dst[a]]
                spd *= m1 if m1 < m2 else m2
            if spd < 1e-6:
                spd = 1e-6
            new_pos = g_pos[g] + spd * (t_end_step - g_clock[g])
            if new_pos >= arc_len[a]:
                arr_g[n_arr] = g
                arr_t[n_arr] = g_clock[g] + (arc_len[a] - g_pos[g]) / spd
                n_arr += 1
            else:
                g_pos[g] = new_pos
                g_clock[g] = t_end_step
        if n_arr > 0:
            order = np.argsort(arr_t[:n_arr])
            for oi in range(n_arr):
                g = arr_g[order[oi]]
                ta = arr_t[order[oi]]
                a = g_arc[g]
                v = arc_dst[a]
                if arc_kind[a] == 1:
                    g_floors[g] += 1.0
                leg = g_leg[g]
                at_wp = leg < g_wp.shape[1] and g_wp[g, leg] >= 0 and wp_target[g_wp[g, leg]] == v
                if node_is_exit[v]:
                    a_occ[a] -= g_space[g]
                    g_state[g] = ST_EXITED
                    out_exit[g] = ta
                    if out_left[g] < 0.0:
                        out_left[g] = ta
                    n_done += 1
                elif at_wp:
                    a_occ[a] -= g_space[g]
                    g_node[g] = v
                    g_arc[g] = -1
                    if g_wp_kind[g, leg] == 0:
                        g_state[g] = ST_DWELL
                        g_ready[g] = ta + g_wp_dwell[g, leg]
                    else:
                        g_state[g] = ST_WAIT_LIFT
                        _push(w_head, w_tail, g_next, v, g)
                        w_space[v] += g_space[g]
                    g_leg[g] = leg + 1
                else:
                    g_state[g] = ST_QUEUED
                    g_node[g] = v
                    g_arrive[g] = ta
                    _push(a_qhead, a_qtail, g_next, a, g)
                    a_qspace[a] += g_space[g]
                    a_qpers[a] += g_count[g]
                    n_demand[v] += 1

        # ------------------------------------------------ 8. toxic exposure
        if hi >= 0:
            for g in range(G):
                s = g_state[g]
                if s == ST_EXITED or s == ST_INCAPACITATED or s == ST_IN_LIFT:
                    continue
                if s == ST_ON_ARC:
                    a = g_arc[g]
                    node = arc_dst[a] if g_pos[g] > 0.5 * arc_len[a] else arc_src[a]
                else:
                    node = g_node[g]
                out_fed[g] += hz_fed[hi, node] * dt
                if out_fed[g] >= fed_threshold:
                    if s == ST_ON_ARC or s == ST_QUEUED:
                        a = g_arc[g]
                        a_occ[a] -= g_space[g]
                        if s == ST_QUEUED:
                            a_qspace[a] -= g_space[g]
                            a_qpers[a] -= g_count[g]
                    elif s == ST_WAIT_LIFT:
                        w_space[g_node[g]] -= g_space[g]
                    elif s == ST_WAIT_RESCUE:
                        n_wait_rescue -= 1
                    g_state[g] = ST_INCAPACITATED
                    n_done += 1

        # ------------------------------------------------ 9. bookkeeping
        for a in range(M):
            q = a_qpers[a]
            if q > out_arc_maxq[a]:
                out_arc_maxq[a] = q
            out_arc_qint[a] += q * dt
        if step % rec_every == 0 and rec < n_rec_max:
            for a in range(M):
                out_series[a, rec] = a_qpers[a]
            rec += 1
        step += 1
        t = step * dt

        # ------------------------------------------------ 10. fast-forward rescue-only tails
        if n_done + n_wait_rescue == G and n_wait_rescue > 0 and team_free.size > 0:
            t_now = t if t > rescue_start else rescue_start
            while n_wait_rescue > 0:
                k = 0
                for kk in range(team_free.size):
                    if team_free[kk] < team_free[k]:
                        k = kk
                start = team_free[k] if team_free[k] > t_now else t_now
                pick = -1
                best_lv = 1 << 30
                for g in range(G):
                    if g_state[g] == ST_WAIT_RESCUE:
                        lv = node_level[g_node[g]]
                        if lv < best_lv:
                            best_lv = lv
                            pick = g
                lv = float(best_lv)
                reach = start + rescue_climb * lv + rescue_handling
                done = reach + rescue_carry * lv
                if out_left[pick] < 0.0:
                    out_left[pick] = reach
                out_exit[pick] = done
                out_rescued[pick] = True
                g_state[pick] = ST_EXITED
                n_wait_rescue -= 1
                n_done += 1
                n_rescued += 1
                team_free[k] = done

    t_end = 0.0
    for g in range(G):
        out_state[g] = g_state[g]
        if out_exit[g] < np.inf and out_exit[g] > t_end:
            t_end = out_exit[g]
    out_meta[META_T_END] = t_end
    out_meta[META_STEPS] = step
    out_meta[META_N_REC] = rec
    out_meta[META_FORCED] = forced
    out_meta[META_LIFT_TRIPS] = lift_trips
    out_meta[META_N_RESCUED] = n_rescued
