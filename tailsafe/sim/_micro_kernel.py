"""Numba kernel of the microscopic replay engine (see :mod:`tailsafe.sim.micro`).

People are discs walking in the rectangles of the floor plan (rooms, corridors,
lobbies, stair landings) with a collision-free speed model: the walking
direction is the direction to the next doorway plus exponential repulsion from
neighbours and walls; the speed is ``min(v0, max(0, gap / T))`` where ``gap``
is the free distance to the nearest person ahead. Rectangles are connected by
*portals* (door openings or shared boundaries); people cross a portal only if
there is room on the other side. Stair flights are strips of ``lanes`` lanes,
entered at a landing's departure point and left at the next landing's arrival
point. Routing decisions reuse the meso engine's next-hop tables.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from tailsafe.sim._kernel import _desired_arc

S_WAIT = 0  # before reacting (in the flat)
S_WALK = 1
S_FLIGHT = 2
S_DWELL = 3  # at a waypoint (e.g. a relative's flat, a refuge floor)
S_EXITED = 4
S_EXCLUDED = 5  # not walked by the micro engine (lift / rescue): taken from meso
S_STRANDED = 6  # no usable route: waits for rescue where it is

META_STEPS = 0
META_FORCED = 1
META_N_REC = 2
META_SIZE = 3

_CUTOFF = 2.5  # m, neighbour interaction range


@njit(cache=True)
def _occupied(px, py, rad, node, x, y, r, st, n_ptr, n_list, skip):  # type: ignore[no-untyped-def]
    """True if a disc of radius ``rad`` at (px, py) overlaps anyone in ``node``."""
    for k in range(n_ptr[node], n_ptr[node + 1]):
        j = n_list[k]
        if j == skip:
            continue
        dx = x[j] - px
        dy = y[j] - py
        lim = 0.9 * (r[j] + rad)
        if dx * dx + dy * dy < lim * lim:
            return True
    return False


@njit(cache=True)
def run_micro_kernel(  # type: ignore[no-untyped-def]
    # ---- geometry
    n_x0,
    n_x1,
    n_y0,
    n_y1,
    node_level,
    node_is_exit,
    node_cap,
    arc_src,
    arc_dst,
    arc_kind,
    arc_len,
    arc_lanes,
    arc_lane_w,
    arc_rev,
    p_x0,
    p_y0,
    p_x1,
    p_y1,
    p_nx,
    p_ny,
    # ---- routing (meso tables)
    next_arc,
    state_time,
    wp_target,
    wp_next,
    arc_blocked_from,
    # ---- people
    a_node0,
    a_x0,
    a_y0,
    a_ready,
    a_vh,
    a_vdown,
    a_vup,
    a_fmin,
    a_fefold,
    a_rad,
    a_class,
    a_wp,
    a_wp_dwell,
    a_wp_kind,
    a_active,
    # ---- hazard
    hz_dt,
    hz_speed,
    # ---- model
    time_gap,
    rep_a,
    rep_d,
    wall_a,
    wall_d,
    dt,
    t_max,
    stuck_time,
    rec_every,
    # ---- outputs
    out_exit,
    out_left,
    out_state,
    out_node,
    out_arc,
    rec_level,
    rec_x,
    rec_y,
    rec_state,
    meta,
):
    A = a_node0.size
    N = n_x0.size
    K = state_time.size
    H = hz_speed.shape[0]
    R = rec_level.shape[0]

    st = np.empty(A, dtype=np.int8)
    node = a_node0.copy()
    arc = np.full(A, -2, dtype=np.int32)
    x = a_x0.copy()
    y = a_y0.copy()
    s = np.zeros(A)
    u = np.zeros(A)
    vx = np.zeros(A)
    vy = np.zeros(A)
    know = np.zeros(A, dtype=np.int32)
    leg = np.zeros(A, dtype=np.int32)
    floors = np.zeros(A)
    blocked = np.zeros(A)
    dwell_until = np.zeros(A)
    home_level = np.empty(A, dtype=np.int32)
    max_lanes = 1
    for b in range(arc_src.size):
        if arc_lanes[b] > max_lanes:
            max_lanes = arc_lanes[b]
    lane_free = np.zeros((arc_src.size, max_lanes))  # doorway lanes: next crossing time
    still = np.zeros(A)  # seconds without getting anywhere
    anc_x = a_x0.copy()
    anc_y = a_y0.copy()
    anc_t = np.zeros(A)
    dirx = np.zeros(A)
    diry = np.zeros(A)
    blocker = np.full(A, -1, dtype=np.int64)
    n_done = 0
    for i in range(A):
        home_level[i] = node_level[a_node0[i]]
        out_exit[i] = np.inf
        out_left[i] = -1.0
        if a_active[i]:
            st[i] = S_WAIT
        else:
            st[i] = S_EXCLUDED
            n_done += 1

    n_cnt = np.zeros(N + 1, dtype=np.int64)
    n_ptr = np.zeros(N + 1, dtype=np.int64)
    n_list = np.empty(A, dtype=np.int64)
    M = arc_src.size
    f_ptr = np.zeros(M + 1, dtype=np.int64)
    f_cnt = np.zeros(M + 1, dtype=np.int64)
    f_list = np.empty(A, dtype=np.int64)

    state = 0
    step = 0
    rec = 0
    forced = 0
    t = 0.0
    while t < t_max and n_done < A:
        while state + 1 < K and state_time[state + 1] <= t:
            state += 1
        hi = -1
        if H > 0:
            hi = int(t / hz_dt)
            if hi >= H:
                hi = H - 1

        # -------------------------------------------- 1. reactions and dwell ends
        for i in range(A):
            if st[i] == S_WAIT and a_ready[i] <= t:
                st[i] = S_WALK
                arc[i] = -2
                anc_t[i] = t
            elif st[i] == S_DWELL and dwell_until[i] <= t:
                st[i] = S_WALK
                arc[i] = -2

        # -------------------------------------------- 2. who is where
        n_cnt[:] = 0
        f_cnt[:] = 0
        for i in range(A):
            si = st[i]
            if si == S_WALK or si == S_WAIT or si == S_DWELL or si == S_STRANDED:
                n_cnt[node[i] + 1] += 1
            elif si == S_FLIGHT:
                f_cnt[arc[i] + 1] += 1
        n_ptr[0] = 0
        for v in range(N):
            n_ptr[v + 1] = n_ptr[v] + n_cnt[v + 1]
        f_ptr[0] = 0
        for b in range(M):
            f_ptr[b + 1] = f_ptr[b] + f_cnt[b + 1]
        n_cnt[:] = 0
        f_cnt[:] = 0
        for i in range(A):
            si = st[i]
            if si == S_WALK or si == S_WAIT or si == S_DWELL or si == S_STRANDED:
                v = node[i]
                n_list[n_ptr[v] + n_cnt[v]] = i
                n_cnt[v] += 1
            elif si == S_FLIGHT:
                b = arc[i]
                f_list[f_ptr[b] + f_cnt[b]] = i
                f_cnt[b] += 1

        # -------------------------------------------- 3. walking velocities
        for i in range(A):
            vx[i] = 0.0
            vy[i] = 0.0
            if st[i] != S_WALK:
                continue
            v = node[i]
            if arc[i] == -2 or (arc[i] >= 0 and arc_blocked_from[arc[i]] <= t):
                b = _desired_arc(
                    i,
                    v,
                    t,
                    state,
                    know,
                    a_class,
                    leg,
                    a_wp,
                    a_wp_kind,
                    wp_target,
                    wp_next,
                    next_arc,
                    arc_blocked_from,
                )
                if b < 0:
                    st[i] = S_STRANDED
                    n_done += 1
                    continue
                arc[i] = b
            b = arc[i]
            ri = a_rad[i]
            if arc_kind[b] == 0:
                # aim at the nearest point of the portal, kept a radius from its ends
                ex = p_x1[b] - p_x0[b]
                ey = p_y1[b] - p_y0[b]
                plen = np.sqrt(ex * ex + ey * ey)
                if plen > 1e-9:
                    ex /= plen
                    ey /= plen
                lo = ri if plen > 2.0 * ri else 0.5 * plen
                hi_ = plen - ri if plen > 2.0 * ri else 0.5 * plen
                proj = (x[i] - p_x0[b]) * ex + (y[i] - p_y0[b]) * ey
                outside = proj < 0.0 or proj > plen
                if proj < lo:
                    proj = lo
                elif proj > hi_:
                    proj = hi_
                if outside:
                    # first get in front of the doorway, then walk through it
                    tx = p_x0[b] + ex * proj - p_nx[b] * (ri + 0.3)
                    ty = p_y0[b] + ey * proj - p_ny[b] * (ri + 0.3)
                else:
                    tx = p_x0[b] + ex * proj + p_nx[b] * 0.5
                    ty = p_y0[b] + ey * proj + p_ny[b] * 0.5
            else:
                # the flight's entrance: a line across the stair at the departure point
                hw = 0.5 * arc_lanes[b] * arc_lane_w[b]
                lat = (x[i] - p_x0[b]) * p_nx[b] + (y[i] - p_y0[b]) * p_ny[b]
                lim = hw - 0.5 * ri if hw > ri else 0.0
                if lat < -lim:
                    lat = -lim
                elif lat > lim:
                    lat = lim
                tx = p_x0[b] + p_nx[b] * lat
                ty = p_y0[b] + p_ny[b] * lat
            dx = tx - x[i]
            dy = ty - y[i]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-9:
                ex0 = 0.0
                ey0 = 0.0
            else:
                ex0 = dx / dist
                ey0 = dy / dist
            # repulsion from neighbours (this room and the room behind the door)
            fx = ex0
            fy = ey0
            gap = 1e9
            for pass_ in range(2):
                w = v
                if pass_ == 1:
                    if arc_kind[b] != 0 or node_is_exit[arc_dst[b]]:
                        break
                    w = arc_dst[b]
                for k in range(n_ptr[w], n_ptr[w + 1]):
                    j = n_list[k]
                    if j == i:
                        continue
                    ddx = x[i] - x[j]
                    ddy = y[i] - y[j]
                    d = np.sqrt(ddx * ddx + ddy * ddy)
                    if d > _CUTOFF:
                        continue
                    if d < 1e-6:
                        ddx = 1e-3 if i > j else -1e-3
                        ddy = 0.0
                        d = 1e-3
                    lij = ri + a_rad[j]
                    f = rep_a * np.exp((lij - d) / rep_d)
                    fx += f * ddx / d
                    fy += f * ddy / d
            # walls of this rectangle (not the one being walked through)
            wx0 = x[i] - n_x0[v] - ri
            wx1 = n_x1[v] - x[i] - ri
            wy0 = y[i] - n_y0[v] - ri
            wy1 = n_y1[v] - y[i] - ri
            skip_w = -1
            if arc_kind[b] == 0:
                if p_nx[b] < -0.5:
                    skip_w = 0
                elif p_nx[b] > 0.5:
                    skip_w = 1
                elif p_ny[b] < -0.5:
                    skip_w = 2
                elif p_ny[b] > 0.5:
                    skip_w = 3
            if skip_w != 0 and wx0 < 0.5:
                fx += wall_a * np.exp(-max(wx0, 0.0) / wall_d)
            if skip_w != 1 and wx1 < 0.5:
                fx -= wall_a * np.exp(-max(wx1, 0.0) / wall_d)
            if skip_w != 2 and wy0 < 0.5:
                fy += wall_a * np.exp(-max(wy0, 0.0) / wall_d)
            if skip_w != 3 and wy1 < 0.5:
                fy -= wall_a * np.exp(-max(wy1, 0.0) / wall_d)
            fn = np.sqrt(fx * fx + fy * fy)
            if fn < 1e-9:
                continue
            fx /= fn
            fy /= fn
            # headway along the direction actually taken
            dirx[i] = fx
            diry[i] = fy
            blocker[i] = -1
            for pass_ in range(2):
                w = v
                if pass_ == 1:
                    if arc_kind[b] != 0 or node_is_exit[arc_dst[b]]:
                        break
                    w = arc_dst[b]
                for k in range(n_ptr[w], n_ptr[w + 1]):
                    j = n_list[k]
                    if j == i:
                        continue
                    ddx = x[j] - x[i]
                    ddy = y[j] - y[i]
                    along = ddx * fx + ddy * fy
                    if along <= 0.0 or along > _CUTOFF:
                        continue
                    lij = ri + a_rad[j]
                    perp = abs(-ddx * fy + ddy * fx)
                    if perp < lij:
                        c = along - np.sqrt(lij * lij - perp * perp)
                        if c < gap:
                            gap = c
                            blocker[i] = j
            v0 = a_vh[i]
            if hi >= 0:
                v0 *= hz_speed[hi, v]
            spd = gap / time_gap
            if spd > v0:
                spd = v0
            if spd < 0.0:
                spd = 0.0
            if spd > dist / dt and arc_kind[b] != 0:
                spd = dist / dt  # don't overshoot a stair departure point
            vx[i] = spd * fx
            vy[i] = spd * fy

        # -------------------------------------------- 4. flight speeds
        for b in range(M):
            if f_ptr[b + 1] == f_ptr[b]:
                continue
            for k in range(f_ptr[b], f_ptr[b + 1]):
                i = f_list[k]
                if arc_kind[b] == 1:
                    fm = a_fmin[i]
                    v0 = a_vdown[i] * (fm + (1.0 - fm) * np.exp(-floors[i] / a_fefold[i]))
                else:
                    v0 = a_vup[i]
                if hi >= 0:
                    m1 = hz_speed[hi, arc_src[b]]
                    m2 = hz_speed[hi, arc_dst[b]]
                    v0 *= m1 if m1 < m2 else m2
                gap = 1e9
                for k2 in range(f_ptr[b], f_ptr[b + 1]):
                    j = f_list[k2]
                    if j == i or s[j] <= s[i]:
                        continue
                    if abs(u[j] - u[i]) < 0.5 * arc_lane_w[b]:
                        c = s[j] - s[i] - (a_rad[i] + a_rad[j])
                        if c < gap:
                            gap = c
                spd = gap / time_gap
                if spd > v0:
                    spd = v0
                if spd < 0.0:
                    spd = 0.0
                vx[i] = spd

        # -------------------------------------------- 5. move and change rooms
        t_end = t + dt
        for i in range(A):
            si = st[i]
            if si == S_WALK:
                v = node[i]
                b = arc[i]
                ri = a_rad[i]
                nxp = x[i] + vx[i] * dt
                nyp = y[i] + vy[i] * dt
                crossed = False
                in_span = False
                if arc_kind[b] == 0:
                    ex = p_x1[b] - p_x0[b]
                    ey = p_y1[b] - p_y0[b]
                    plen = np.sqrt(ex * ex + ey * ey)
                    if plen > 1e-9:
                        ex /= plen
                        ey /= plen
                    proj = (nxp - p_x0[b]) * ex + (nyp - p_y0[b]) * ey
                    dn = (nxp - p_x0[b]) * p_nx[b] + (nyp - p_y0[b]) * p_ny[b]
                    in_span = proj >= 0.0 and proj <= plen
                    # Doorway lanes: one person per lane per headway (T + body / speed).
                    lane = int(proj / plen * arc_lanes[b]) if plen > 1e-9 else 0
                    if lane < 0:
                        lane = 0
                    elif lane >= arc_lanes[b]:
                        lane = arc_lanes[b] - 1
                    if in_span and dn > -ri and lane_free[b, lane] > t:
                        blocked[i] += dt
                    elif in_span and dn > -ri:
                        w = arc_dst[b]
                        cx = p_x0[b] + ex * proj
                        cy = p_y0[b] + ey * proj
                        head = time_gap + 2.0 * ri / max(a_vh[i], 0.1)
                        if node_is_exit[w]:
                            lane_free[b, lane] = t_end + head
                            st[i] = S_EXITED
                            out_exit[i] = t_end
                            if out_left[i] < 0.0:
                                out_left[i] = t_end
                            n_done += 1
                            crossed = True
                        else:
                            ix = cx + p_nx[b] * (ri + 0.05)
                            iy = cy + p_ny[b] * (ri + 0.05)
                            free = not _occupied(ix, iy, ri, w, x, y, a_rad, st, n_ptr, n_list, i)
                            force = blocked[i] >= stuck_time and n_cnt[w] < node_cap[w]
                            if free or force:
                                if not free:
                                    forced += 1
                                lane_free[b, lane] = t_end + head
                                n_cnt[v] -= 1
                                n_cnt[w] += 1
                                node[i] = w
                                x[i] = ix
                                y[i] = iy
                                arc[i] = -2
                                blocked[i] = 0.0
                                crossed = True
                            else:
                                blocked[i] += dt
                    # Head-on at a doorway: squeeze past someone coming the other way.
                    if not crossed and dn > -(ri + 0.5) and in_span and arc_rev[b] >= 0:
                        w = arc_dst[b]
                        if not node_is_exit[w]:
                            cx = p_x0[b] + ex * min(max(proj, 0.0), plen)
                            cy = p_y0[b] + ey * min(max(proj, 0.0), plen)
                            for k in range(n_ptr[w], n_ptr[w + 1]):
                                j = n_list[k]
                                if st[j] != S_WALK or node[j] != w or arc[j] != arc_rev[b]:
                                    continue
                                rj = a_rad[j]
                                ddx = x[j] - cx
                                ddy = y[j] - cy
                                if ddx * ddx + ddy * ddy > (rj + 0.5) * (rj + 0.5):
                                    continue
                                node[i] = w
                                x[i] = cx + p_nx[b] * (ri + 0.05)
                                y[i] = cy + p_ny[b] * (ri + 0.05)
                                arc[i] = -2
                                blocked[i] = 0.0
                                node[j] = v
                                x[j] = cx - p_nx[b] * (rj + 0.05)
                                y[j] = cy - p_ny[b] * (rj + 0.05)
                                arc[j] = -2
                                blocked[j] = 0.0
                                crossed = True
                                break
                elif arc_kind[b] != 0:
                    hw = 0.5 * arc_lanes[b] * arc_lane_w[b]
                    lat = (nxp - p_x0[b]) * p_nx[b] + (nyp - p_y0[b]) * p_ny[b]
                    lat_c = min(max(lat, -hw), hw)
                    ddx = p_x0[b] + p_nx[b] * lat_c - nxp
                    ddy = p_y0[b] + p_ny[b] * lat_c - nyp
                    reach = ri + 0.3
                    if ddx * ddx + ddy * ddy < reach * reach:
                        # enter the flight: own lane if free, else the lane with most room
                        lanes = arc_lanes[b]
                        own = int((lat_c + hw) / arc_lane_w[b])
                        if own >= lanes:
                            own = lanes - 1
                        best_lane = -1
                        best_room = -1.0
                        for ln in range(lanes):
                            uc = (ln + 0.5) * arc_lane_w[b]
                            room = 1e9
                            for k in range(f_ptr[b], f_ptr[b + 1]):
                                j = f_list[k]
                                if st[j] == S_FLIGHT and abs(u[j] - uc) < 0.5 * arc_lane_w[b]:
                                    if s[j] < room:
                                        room = s[j]
                            if ln == own and room >= ri + 0.2:
                                room = 2e9  # prefer the lane in front of you
                            if room > best_room:
                                best_room = room
                                best_lane = ln
                        ok = best_room >= ri + 0.2
                        if ok:
                            n_cnt[v] -= 1
                            st[i] = S_FLIGHT
                            s[i] = 0.0
                            u[i] = (best_lane + 0.5) * arc_lane_w[b]
                            blocked[i] = 0.0
                            if out_left[i] < 0.0 and node_level[arc_dst[b]] != home_level[i]:
                                out_left[i] = t_end
                            crossed = True
                        else:
                            blocked[i] += dt
                j = blocker[i]
                if (
                    not crossed
                    and still[i] > 5.0
                    and j >= 0
                    and st[j] == S_WALK
                    and node[j] == v
                    and still[j] > 5.0
                    and dirx[i] * dirx[j] + diry[i] * diry[j] < 0.0
                ):
                    # Stand-off between two people facing each other: squeeze past.
                    tx = x[j]
                    ty = y[j]
                    x[j] = x[i]
                    y[j] = y[i]
                    x[i] = tx
                    y[i] = ty
                    still[i] = 0.0
                    still[j] = 0.0
                    anc_x[i] = x[i]
                    anc_y[i] = y[i]
                    anc_t[i] = t_end
                    anc_x[j] = x[j]
                    anc_y[j] = y[j]
                    anc_t[j] = t_end
                    continue
                if not crossed:
                    # stay inside the rectangle, except towards the doorway
                    lo_x = n_x0[v] + ri
                    hi_x = n_x1[v] - ri
                    lo_y = n_y0[v] + ri
                    hi_y = n_y1[v] - ri
                    if arc_kind[b] == 0 and in_span:
                        if p_nx[b] < -0.5:
                            lo_x = n_x0[v] - ri
                        elif p_nx[b] > 0.5:
                            hi_x = n_x1[v] + ri
                        elif p_ny[b] < -0.5:
                            lo_y = n_y0[v] - ri
                        elif p_ny[b] > 0.5:
                            hi_y = n_y1[v] + ri
                    if lo_x > hi_x:
                        lo_x = hi_x = 0.5 * (n_x0[v] + n_x1[v])
                    if lo_y > hi_y:
                        lo_y = hi_y = 0.5 * (n_y0[v] + n_y1[v])
                    x[i] = min(max(nxp, lo_x), hi_x)
                    y[i] = min(max(nyp, lo_y), hi_y)
                    mdx = x[i] - anc_x[i]
                    mdy = y[i] - anc_y[i]
                    if mdx * mdx + mdy * mdy > 0.09:
                        anc_x[i] = x[i]
                        anc_y[i] = y[i]
                        anc_t[i] = t_end
                    still[i] = t_end - anc_t[i]
                elif st[i] == S_WALK:
                    # arrived in a new room: a waypoint?
                    w = node[i]
                    lg = leg[i]
                    if lg < a_wp.shape[1] and a_wp[i, lg] >= 0 and wp_target[a_wp[i, lg]] == w:
                        leg[i] = lg + 1
                        if a_wp_kind[i, lg] == 0:
                            st[i] = S_DWELL
                            dwell_until[i] = t_end + a_wp_dwell[i, lg]
            elif si == S_FLIGHT:
                b = arc[i]
                s[i] += vx[i] * dt
                if s[i] >= arc_len[b]:
                    s[i] = arc_len[b]
                    w = arc_dst[b]
                    ri = a_rad[i]
                    # arrival point, offset across the landing by the lane used
                    ix = p_x1[b] + (u[i] - 0.5 * arc_lanes[b] * arc_lane_w[b]) * p_nx[b]
                    iy = p_y1[b] + (u[i] - 0.5 * arc_lanes[b] * arc_lane_w[b]) * p_ny[b]
                    ix = min(max(ix, n_x0[w] + ri), n_x1[w] - ri)
                    iy = min(max(iy, n_y0[w] + ri), n_y1[w] - ri)
                    free = not _occupied(ix, iy, ri, w, x, y, a_rad, st, n_ptr, n_list, i)
                    force = blocked[i] >= stuck_time and n_cnt[w] < node_cap[w]
                    if free or force:
                        if not free:
                            forced += 1
                            # step in beside whoever stands on the arrival point
                            ix += 0.3 * ((i % 3) - 1) * p_nx[b] + 0.05 * ((i % 2) * 2 - 1)
                            iy += 0.3 * ((i % 3) - 1) * p_ny[b]
                            ix = min(max(ix, n_x0[w] + ri), n_x1[w] - ri)
                            iy = min(max(iy, n_y0[w] + ri), n_y1[w] - ri)
                        n_cnt[w] += 1
                        if arc_kind[b] == 1:
                            floors[i] += 1.0
                        st[i] = S_WALK
                        node[i] = w
                        x[i] = ix
                        y[i] = iy
                        arc[i] = -2
                        blocked[i] = 0.0
                        lg = leg[i]
                        if lg < a_wp.shape[1] and a_wp[i, lg] >= 0 and wp_target[a_wp[i, lg]] == w:
                            leg[i] = lg + 1
                            if a_wp_kind[i, lg] == 0:
                                st[i] = S_DWELL
                                dwell_until[i] = t_end + a_wp_dwell[i, lg]
                    else:
                        blocked[i] += dt

        # -------------------------------------------- 6. no stacked discs
        # People who ended up overlapping (forced moves, simultaneous entries)
        # are pushed apart by part of the overlap, inside their rectangle.
        for v in range(N):
            lo = n_ptr[v]
            hi_ = n_ptr[v + 1]
            if hi_ - lo < 2:
                continue
            for k in range(lo, hi_):
                i = n_list[k]
                if st[i] != S_WALK or node[i] != v:
                    continue
                for k2 in range(k + 1, hi_):
                    j = n_list[k2]
                    if node[j] != v or (st[j] != S_WALK and st[j] != S_WAIT and st[j] != S_DWELL):
                        continue
                    ddx = x[i] - x[j]
                    ddy = y[i] - y[j]
                    d = np.sqrt(ddx * ddx + ddy * ddy)
                    lij = 0.9 * (a_rad[i] + a_rad[j])
                    if d >= lij:
                        continue
                    if d < 1e-6:
                        ddx = 1.0
                        ddy = 0.0
                        d = 1.0
                        push = 0.25 * lij
                    else:
                        push = 0.25 * (lij - d)
                    ux = ddx / d
                    uy = ddy / d
                    ri = a_rad[i]
                    x[i] = min(max(x[i] + ux * push, n_x0[v] + ri), n_x1[v] - ri)
                    y[i] = min(max(y[i] + uy * push, n_y0[v] + ri), n_y1[v] - ri)
                    if st[j] == S_WALK:
                        rj = a_rad[j]
                        x[j] = min(max(x[j] - ux * push, n_x0[v] + rj), n_x1[v] - rj)
                        y[j] = min(max(y[j] - uy * push, n_y0[v] + rj), n_y1[v] - rj)

        # -------------------------------------------- 7. record
        if step % rec_every == 0 and rec < R:
            for i in range(A):
                si = st[i]
                rec_state[rec, i] = si
                if si == S_EXITED:
                    rec_level[rec, i] = -1
                    continue
                if si == S_FLIGHT:
                    # Drawn on the upper level, walking across the stair
                    # enclosure from the departure point to the far end.
                    b = arc[i]
                    f = s[i] / arc_len[b] if arc_len[b] > 0 else 0.0
                    if arc_kind[b] == 1:
                        up = arc_src[b]
                        sx = p_x0[b]
                        sy = p_y0[b]
                    else:
                        up = arc_dst[b]
                        sx = p_x1[b]
                        sy = p_y1[b]
                        f = 1.0 - f
                    cx = 0.5 * (n_x0[up] + n_x1[up])
                    cy = 0.5 * (n_y0[up] + n_y1[up])
                    off = u[i] - 0.5 * arc_lanes[b] * arc_lane_w[b]
                    rec_level[rec, i] = node_level[up]
                    rec_x[rec, i] = sx + (cx - sx) * 2.0 * f + off * p_nx[b]
                    rec_y[rec, i] = sy + (cy - sy) * 2.0 * f + off * p_ny[b]
                else:
                    rec_level[rec, i] = node_level[node[i]]
                    rec_x[rec, i] = x[i]
                    rec_y[rec, i] = y[i]
            rec += 1
        step += 1
        t = t_end

    for i in range(A):
        out_state[i] = st[i]
        out_node[i] = node[i] if st[i] != S_FLIGHT else -1 - arc[i]
        out_arc[i] = arc[i]
    meta[META_STEPS] = step
    meta[META_FORCED] = forced
    meta[META_N_REC] = rec


@njit(cache=True)
def periodic_corridor(  # type: ignore[no-untyped-def]
    x0, y0, rad, v0, length, width, time_gap, rep_a, rep_d, wall_a, wall_d, dt, t_warm, t_meas
):
    """Mean walking speed in a periodic corridor (fundamental-diagram check).

    Same direction and speed rules as the replay kernel, everyone heading in
    +x, walls at y = 0 and y = ``width``, periodic in x.
    """
    A = x0.size
    x = x0.copy()
    y = y0.copy()
    vx = np.zeros(A)
    vy = np.zeros(A)
    t = 0.0
    dist = 0.0
    n_meas = 0
    while t < t_warm + t_meas:
        for i in range(A):
            fx = 1.0
            fy = 0.0
            for j in range(A):
                if j == i:
                    continue
                ddx = x[i] - x[j]
                if ddx > 0.5 * length:
                    ddx -= length
                elif ddx < -0.5 * length:
                    ddx += length
                ddy = y[i] - y[j]
                d = np.sqrt(ddx * ddx + ddy * ddy)
                if d > _CUTOFF:
                    continue
                if d < 1e-6:
                    ddx = 1e-3 if i > j else -1e-3
                    ddy = 0.0
                    d = 1e-3
                f = rep_a * np.exp((rad[i] + rad[j] - d) / rep_d)
                fx += f * ddx / d
                fy += f * ddy / d
            wy0 = y[i] - rad[i]
            wy1 = width - y[i] - rad[i]
            if wy0 < 0.5:
                fy += wall_a * np.exp(-max(wy0, 0.0) / wall_d)
            if wy1 < 0.5:
                fy -= wall_a * np.exp(-max(wy1, 0.0) / wall_d)
            fn = np.sqrt(fx * fx + fy * fy)
            fx /= fn
            fy /= fn
            gap = 1e9
            for j in range(A):
                if j == i:
                    continue
                ddx = x[j] - x[i]
                if ddx > 0.5 * length:
                    ddx -= length
                elif ddx < -0.5 * length:
                    ddx += length
                ddy = y[j] - y[i]
                along = ddx * fx + ddy * fy
                if along <= 0.0 or along > _CUTOFF:
                    continue
                lij = rad[i] + rad[j]
                perp = abs(-ddx * fy + ddy * fx)
                if perp < lij:
                    c = along - np.sqrt(lij * lij - perp * perp)
                    if c < gap:
                        gap = c
            spd = gap / time_gap
            if spd > v0[i]:
                spd = v0[i]
            if spd < 0.0:
                spd = 0.0
            vx[i] = spd * fx
            vy[i] = spd * fy
        for i in range(A):
            x[i] = (x[i] + vx[i] * dt) % length
            y[i] = min(max(y[i] + vy[i] * dt, rad[i]), width - rad[i])
            if t >= t_warm:
                dist += vx[i] * dt
        if t >= t_warm:
            n_meas += 1
        t += dt
    return dist / max(n_meas, 1) / dt / A
