"""Apex Tactician 23: wider restarts and bounded dribble evaluation.

Uses only the public soccer SDK and Python's standard library. Close games use full tactical
calculations each tick; distant opponents are pruned only where they
cannot affect the original capped clearance and danger scores.
Designed for the documented five-a-side, ground-play course profile.
"""
import math
from soccer import TeamController, clamp

_hypot = math.hypot
_cos = math.cos
_sin = math.sin
_sqrt = math.sqrt
_log = math.log

PATH_TICKS = 62
CHASER_HYSTERESIS = 1.6
PASS_LOCKOUT = 2
MARK_GAP = 2.4
SHOT_RANGE = 22.0
SHOT_MARGIN = -0.35
PRESSURE_RADIUS = 3.5
PRESSURE_RADIUS_SQ = PRESSURE_RADIUS * PRESSURE_RADIUS
N_DEF_OURS = 2
N_DEF_THEIRS = 3
SCREEN_DIST = 8.0
SCREEN_DIST_OURS = 40.0


def _action(movement=(0.0, 0.0), kick_direction=None, kick_power=0.0):
    if kick_direction is None:
        return {'movement': movement}
    return {
        'movement': movement,
        'kick_direction': kick_direction,
        'kick_power': kick_power,
    }


class _LazyPath:
    def __init__(self, source, last_sample):
        self.source = iter(source)
        self.points = []
        self.last_sample = last_sample

    def __iter__(self):
        yield from self.points
        for point in self.source:
            self.points.append(point)
            yield point


class MyTeam(TeamController):
    name = 'Apex Tactician'
    version = '23.0-points'

    def __init__(self):
        self.reset(0)

    def reset(self, seed):
        self._ready = False
        self._cached_movement = None
        self.kick_team = None
        self.chaser = -1
        self.last_pass_kick = {}
        self.targets = {}
        self.defenders = set()
        self._restart_score = None
        self._restart_until = -1

    def initial_formation(self, field):
        keeper_x = field.my_goal[0] + field.player_radius * 2.0
        r = field.centre_circle_radius
        # A central anchor protects the first attack; two advanced wide
        # outlets give the kickoff taker useful forward passing options.
        return [
            (keeper_x, 0.0),
            (-24.0, 0.0),
            (-r - 0.35, -2.5),
            (-0.1, -12.0),
            (-0.1, 12.0),
        ]

    def _setup(self, obs):
        field = obs.field
        hz = getattr(field, 'simulation_hz', None)
        if hz is None:
            hz = (
                obs.simulation_hz()
                if callable(getattr(obs, 'simulation_hz', None))
                else 20.0
            )
        self.half_w = field.width * 0.5
        self.half_h = field.height * 0.5
        self.gw = field.goal_width * 0.5
        self.vmax = field.max_speed
        self.acc = field.acceleration
        self.dt = 1.0 / float(hz)
        self.krange = field.kick_range
        self._per_tick = self.vmax * self.dt
        self._control_range = self.krange * 0.92
        self._turn_ticks = 0.45 / (self.acc * self.dt)
        self._steer_response = self.acc * self.dt * (1.0 - 0.6561) / 0.1
        self.fric = getattr(field, 'ball_friction', 0.985)
        self.dribble_roll = (
            self.dt * (1.0 - self.fric ** 5) / (1.0 - self.fric)
            if self.fric != 1.0
            else 5.0 * self.dt
        )
        self.circ = field.centre_circle_radius
        self.rad = field.player_radius
        self.ball_rad = getattr(field, 'ball_radius', 0.45)
        self.body_radius = self.rad + self.ball_rad + 0.05
        self.impulse = field.kick_impulse
        self._reach_sq = tuple(
            (self.vmax * self.dt * k + self.krange + 1e-09) ** 2
            for k in range(PATH_TICKS)
        )
        self.goal_x = -self.half_w
        self.shape = self.initial_formation(field)
        post_in = self.gw - 0.65
        mid_in = self.gw * 0.48
        self.shot_ys = (-post_in, post_in, -mid_in, mid_in, 0.0)
        self.dribble_angles = tuple(
            (
                ang,
                _cos(ang),
                _sin(ang),
                11.7 + 5.5 * _cos(ang) - 0.35 * abs(ang),
            )
            for ang in (
                0.0,
                0.16, -0.16,
                0.32, -0.32,
                0.48, -0.48,
                0.64, -0.64,
                0.8, -0.8,
                0.96, -0.96,
                1.12, -1.12,
                1.28, -1.28,
                1.44, -1.44,
                1.6, -1.6,
                1.76, -1.76,
                1.92, -1.92,
                2.08, -2.08,
                2.24, -2.24,
                2.4, -2.4,
                2.56, -2.56,
                2.72, -2.72,
                2.88, -2.88,
                3.04, -3.04,
                3.14,
            )
        )
        self._path_roll4 = self.dt * (
            1 + self.fric + self.fric ** 2 + self.fric ** 3
        )
        self._path_decay4 = self.fric ** 4
        count = len(obs.my_players)
        self._default_order = tuple(range(count))
        self._chase_orders = {
            i: (i,) + tuple(j for j in range(count) if j != i)
            for i in range(count)
        }
        self._pass_drag = (1.0 - self.fric) / self.dt
        self._log_fric = _log(self.fric)
        self._ready = True

    def act(self, obs):
        # With a three-goal cushion, uncontested attacking movement may be
        # reused for at most two ticks (0.1 s). Kicks, turnovers, scoring,
        # a playable ball and defensive transitions force an immediate plan.
        if obs.score[0] - obs.score[1] < 3:
            self._cached_movement = None
            return self._act_full(obs)
        if (self._ready and obs.tick % 3 and self._cached_movement is not None
                and tuple(obs.score) == self._restart_score
                and obs.ball.controlling_team == 0
                and obs.ball.velocity[0] >= 0.0
                and not any(e.get('kind') in ('kick', 'turnover', 'goal')
                            for e in obs.events)
                and not any(obs.can_kick(p.id) for p in obs.my_players)):
            return self._cached_movement
        result = self._act_full(obs)
        self._cached_movement = {
            i: {'movement': a['movement']} for i, a in result.items()
        }
        return result

    def _act_full(self, obs):
        if not self._ready:
            self._setup(obs)

        tick = obs.tick
        ball = obs.ball
        bx, by = ball.position
        bvx, bvy = ball.velocity
        me = obs.my_players
        opps = obs.opponents
        opp_xy = [o.position for o in opps]
        ours = ball.controlling_team == 0
        d2c = bx * bx + by * by
        bspeed_sq = bvx * bvx + bvy * bvy
        score = tuple(obs.score)

        if score != self._restart_score:
            self._restart_score = score
            self.kick_team = None
            self._restart_until = tick + int(round(3.0 / self.dt))

        if d2c >= 1.2 or bspeed_sq >= 0.25:
            self._restart_until = tick

        if (
            tick < self._restart_until
            and not ours
            and d2c < 1.2
            and bspeed_sq < 0.25
        ):
            return self._hold_shape(obs)

        for event in obs.events:
            if event.get('kind') == 'kick':
                self.kick_team = event['team']

        if self.kick_team is not None:
            ours = self.kick_team == 0

        path = None
        path_speed = _sqrt(bspeed_sq)
        path_in_bounds = (
            abs(bx) <= self.half_w - self.ball_rad
            and abs(by) <= self.half_h - self.ball_rad
        )
        carrier = (
            opps[min(
                range(len(opps)),
                key=lambda i: (
                    (opp_xy[i][0] - bx) ** 2
                    + (opp_xy[i][1] - by) ** 2
                ),
            )]
            if opps and not ours
            else None
        )

        if carrier is not None and not ours:
            cx, cy = carrier.position
            cvx, cvy = carrier.velocity
            if (
                (cx - bx) ** 2 + (cy - by) ** 2 < 9.0
                and bvx * bvx + bvy * bvy < 156.25
                and (bvx - cvx) ** 2 + (bvy - cvy) ** 2 < 36.0
            ):
                vx, vy = cvx, cvy
                path_speed = _hypot(vx, vy)
                path = _LazyPath(
                    (
                        (
                            k,
                            clamp(
                                bx + vx * self.dt * k,
                                -self.half_w + self.ball_rad,
                                self.half_w - self.ball_rad,
                            ),
                            clamp(
                                by + vy * self.dt * k,
                                -self.half_h + self.ball_rad,
                                self.half_h - self.ball_rad,
                            ),
                        )
                        for k in (
                            0, 1, 2, 3, 4, 5, 6, 7, 8,
                            12, 16, 20, 24, 28, 32, 36,
                            40, 44, 48, 52, 56, 60,
                        )
                    ),
                    60,
                )

        if path is None:
            path = self._ball_path(bx, by, bvx, bvy)

        keeper_may_chase = bx < self.goal_x + 15.0 and abs(by) < self.gw + 9.0
        etas = {}
        best_eta = float('inf')
        best_id = -1
        incumbent_limit = float('inf')

        ordered = self._chase_orders.get(self.chaser, self._default_order)
        for index in ordered:
            p = me[index]
            pid = p.id
            if pid == 0 and not keeper_may_chase:
                continue

            if tick - self.last_pass_kick.get(pid, -999) <= PASS_LOCKOUT:
                if not (
                    bx > self.half_w - 20.0
                    and abs(by) < self.gw + 7.0
                ):
                    continue

            limit = (
                min(best_eta, incumbent_limit)
                if pid != self.chaser
                else best_eta + CHASER_HYSTERESIS
            )

            if limit < 60 and path_in_bounds:
                reach = (
                    self.krange
                    + (self.vmax + path_speed) * self.dt * limit
                    + 1e-9
                )
                px, py = p.position
                if (px - bx) ** 2 + (py - by) ** 2 > reach * reach:
                    continue

            candidate = self._eta(p, path, limit)
            if candidate is not None:
                etas[pid] = candidate
                if (
                    candidate[0] < best_eta
                    or (candidate[0] == best_eta and pid < best_id)
                ):
                    best_eta = candidate[0]
                    best_id = pid

                if pid == self.chaser:
                    incumbent_limit = candidate[0] - CHASER_HYSTERESIS
                    if incumbent_limit <= 0.0:
                        break

        if not etas:
            for p in me:
                etas[p.id] = self._eta(p, path)
            best_id = min(etas, key=lambda i: (etas[i][0], i))

        chaser = best_id
        if (
            self.chaser in etas
            and etas[self.chaser][0]
            <= etas[best_id][0] + CHASER_HYSTERESIS
        ):
            chaser = self.chaser

        self.chaser = chaser
        chase_pt = etas[chaser][1]
        others = [p for p in me if p.id != 0 and p.id != chaser]

        if obs.score[0] > obs.score[1]:
            n_def = 2
        elif ours:
            n_def = N_DEF_OURS
        else:
            n_def = 2

        if not ours and bx < 50:
            n_def = N_DEF_THEIRS
        if len(others) > 3:
            n_def += 1

        others.sort(
            key=lambda p: (
                p.position[0]
                - (5.0 if p.id in self.defenders else 0.0)
            )
        )
        defenders = others[:n_def]
        self.defenders = {p.id for p in defenders}

        mark_spots = self._hybrid_defense(
            defenders, opps, bx, by, bvx, bvy, ours,
            carrier.id if carrier is not None else None,
        )

        if chaser == 0 and defenders:
            cover_spot = self._keeper_spot(bx, by, bvx, bvy)
            cover_id = min(
                defenders,
                key=lambda p: (
                    (p.position[0] - cover_spot[0]) ** 2
                    + (p.position[1] - cover_spot[1]) ** 2
                ),
            ).id
            mark_spots[cover_id] = cover_spot

        actions = {}
        kicked = False
        taken = []
        chaser_action = None
        p = me[chaser]

        if obs.can_kick(chaser):
            chaser_action, is_pass = self._on_ball(
                obs, p, opps, opp_xy, chaser == 0
            )
            if is_pass:
                self.last_pass_kick[chaser] = tick
            kicked = True

        for p in me:
            pid = p.id
            px, py = p.position

            if pid == chaser and chaser_action is not None:
                actions[pid] = chaser_action
                continue

            if not kicked and pid != chaser and obs.can_kick(pid):
                act_ball, is_pass = self._on_ball(
                    obs, p, opps, opp_xy, pid == 0
                )
                if is_pass:
                    self.last_pass_kick[pid] = tick
                actions[pid] = act_ball
                kicked = True
                continue

            if pid == 0 and pid != chaser:
                move = self._keeper_spot(bx, by, bvx, bvy)

            elif pid == chaser:
                tx, ty = chase_pt
                direction_x = self.half_w - tx
                direction_y = -ty
                distance = _hypot(direction_x, direction_y) or 1.0
                move = (
                    tx - direction_x / distance * 1.2,
                    ty - direction_y / distance * 1.2,
                )
                if not ours and bx < 0:
                    move = chase_pt
                if _hypot(bx - px, by - py) < self.krange + 0.5:
                    move = (
                        bx + bvx * self.dt,
                        by + bvy * self.dt,
                    )

            elif pid in self.defenders:
                move = mark_spots.get(pid)
                if move is None:
                    move = (
                        max(self.goal_x + 7.0, min(bx - 12.0, -4.0)),
                        clamp(by * 0.5, -16.0, 16.0),
                    )

            else:
                move = self._space_target(
                    p, opp_xy, bx, by, ours, taken, tick
                )
                taken.append(move)

            mx, my = self._steer(
                px, py, move[0], move[1],
                p.velocity, opp_xy, pid == chaser,
            )
            actions[pid] = {'movement': (mx, my)}

        return actions

    def _hold_shape(self, obs):
        actions = {}
        for p in obs.my_players:
            px, py = p.position
            tx, ty = (
                self.shape[p.id]
                if p.id < len(self.shape)
                else (px, py)
            )
            mx, my = self._steer_simple(px, py, tx, ty)
            mx, my = self._avoid_circle(px, py, mx, my)
            actions[p.id] = {'movement': (mx, my)}
        return actions

    def _avoid_circle(self, px, py, mx, my):
        r = self.circ + 1.15
        step = self.vmax * self.dt * 2.5
        nx, ny = px + mx * step, py + my * step
        if (
            nx * nx + ny * ny >= r * r
            and px * px + py * py >= r * r
        ):
            return mx, my

        d = _hypot(px, py)
        if d < 1e-06:
            return -1.0, 0.0

        rx, ry = px / d, py / d
        if d < r:
            return rx, ry

        tx, ty = -ry, rx
        if tx * mx + ty * my < 0.0:
            tx, ty = -tx, -ty
        tx, ty = tx * 0.6 + rx * 0.4, ty * 0.6 + ry * 0.4
        n = _hypot(tx, ty) or 1.0
        return tx / n, ty / n

    def _ball_path(self, x, y, vx, vy):
        if vx * vx + vy * vy < 0.04:
            return [(0, x, y)]
        return _LazyPath(self._roll_path(x, y, vx, vy), 60)

    def _roll_path(self, x, y, vx, vy):
        dt, f = self.dt, self.fric
        hh = self.half_h - self.ball_rad
        hw = self.half_w - self.ball_rad
        yield 0, x, y
        k = 0

        while k < 60:
            step = 1 if k < 8 else 4
            roll = dt if step == 1 else self._path_roll4
            decay = f if step == 1 else self._path_decay4
            nx, ny = x + vx * roll, y + vy * roll

            if abs(nx) <= hw and abs(ny) <= hh:
                x, y = nx, ny
                vx *= decay
                vy *= decay
            else:
                for _ in range(step):
                    x += vx * dt
                    y += vy * dt
                    vx *= f
                    vy *= f

                    if y > hh:
                        y, vy = 2 * hh - y, -vy * 0.75
                    elif y < -hh:
                        y, vy = -2 * hh - y, -vy * 0.75

                    if x > hw:
                        x, vx = 2 * hw - x, -vx * 0.75
                    elif x < -hw:
                        x, vx = -2 * hw - x, -vx * 0.75

            k += step
            yield k, x, y

    def _eta(self, p, path, limit=float('inf')):
        px, py = p.position
        vx, vy = p.velocity
        vmax = self.vmax
        per_tick = self._per_tick
        kr = self._control_range
        k_over_acc = self._turn_ticks
        cd = p.kick_cooldown_ticks
        t = 0.0
        k = 0
        bx = by = 0.0
        last_sample = (
            path.last_sample
            if isinstance(path, _LazyPath)
            else path[-1][0]
        )
        reach_sq = self._reach_sq

        for k, bx, by in path:
            if k > limit:
                return None
            if k < cd and k != last_sample:
                continue

            dx, dy = bx - px, by - py
            if (
                k != last_sample
                and dx * dx + dy * dy > reach_sq[k]
            ):
                continue

            d = _hypot(dx, dy)
            if d <= kr:
                return float(max(k, cd)), (bx, by)

            u = (vx * dx + vy * dy) / d
            u = -vmax if u < -vmax else vmax if u > vmax else u
            t = (d - kr) / per_tick + (vmax - u) * k_over_acc

            if t <= k:
                return float(max(k, cd)), (bx, by)

        return max(float(cd), t, float(k)), (bx, by)

    def _steer_simple(self, px, py, tx, ty):
        dx, dy = tx - px, ty - py
        d = _hypot(dx, dy)
        if d < 1e-06:
            return 0.0, 0.0
        s = d * 0.55 if d < 1.8 else 1.0
        return dx / d * s, dy / d * s

    def _steer(self, px, py, tx, ty, vel, opp_xy, is_chaser):
        dx, dy = tx - px, ty - py
        d = _hypot(dx, dy)
        if d < 1e-06:
            return 0.0, 0.0
        ux, uy = dx / d, dy / d

        if not (is_chaser and d < 3.0) and d > 1.5:
            rep_x = rep_y = 0.0
            for ox, oy in opp_xy:
                odx, ody = px - ox, py - oy
                if (
                    odx > 2.5 or odx < -2.5
                    or ody > 2.5 or ody < -2.5
                ):
                    continue

                od_sq = odx * odx + ody * ody
                if 1e-06 < od_sq < 6.25:
                    if (ox - px) * ux + (oy - py) * uy > -0.5:
                        inv = 1.0 / od_sq
                        rep_x += odx * inv
                        rep_y += ody * inv

            if rep_x or rep_y:
                ux += rep_x * 0.65
                uy += rep_y * 0.65
                un = _hypot(ux, uy) or 1.0
                ux, uy = ux / un, uy / un

        speed = self.vmax if is_chaser else min(self.vmax, d * 4.4)
        retention = 0.6561
        response = self._steer_response
        mx = (ux * speed - vel[0] * retention) / response
        my = (uy * speed - vel[1] * retention) / response
        size = _hypot(mx, my)
        if size > 1.0:
            mx /= size
            my /= size
        return mx, my

    def _keeper_spot(self, bx, by, bvx, bvy):
        kx = self.goal_x + 3.5
        lim = self.gw - 0.65

        if (
            bvx < -1.0
            and bvx * bvx + bvy * bvy > 196.0
            and bx > kx
        ):
            reaches = (
                -bvx * self.dt
                > (bx - kx) * (1.0 - self.fric)
            )
            if reaches:
                goal_plane = self.goal_x + self.ball_rad
                goal_y = by + bvy * (goal_plane - bx) / bvx
                if abs(goal_y) <= self.gw + 1.5:
                    t = (kx - bx) / bvx
                    return kx, clamp(by + bvy * t, -lim, lim)

        target_y = by * (2.2 / max(4.0, bx - self.goal_x)) * 2.2
        return kx, clamp(target_y, -lim, lim)

    def lane_clearance(self, ax, ay, bx, by, opp_xy):
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return 0.0

        inv_len_sq = 1.0 / length_sq
        best_sq = 1e+18
        for ox, oy in opp_xy:
            t = ((ox - ax) * dx + (oy - ay) * dy) * inv_len_sq
            if t < 0.0:
                continue
            if t > 1.0:
                t = 1.0

            cx, cy = ax + dx * t, ay + dy * t
            d_sq = (ox - cx) ** 2 + (oy - cy) ** 2
            if d_sq < best_sq:
                best_sq = d_sq

        return _sqrt(best_sq) - self.rad

    def _shot_clearance(self, ax, ay, tx, ty, opps, bvx, bvy):
        dx, dy = tx - ax, ty - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            return -10.0

        inv_len_sq = 1.0 / length_sq
        length = _sqrt(length_sq)
        ux, uy = dx / length, dy / length
        cross = bvx * uy - bvy * ux
        budget = self.impulse * self.impulse - cross * cross

        if budget < 0.0:
            return -10.0

        ball_speed = min(
            30.0,
            bvx * ux + bvy * uy + _sqrt(budget),
        )
        if ball_speed < 4.0:
            return -10.0

        best_margin = 1000000000.0
        for o in opps:
            ox, oy = o.position
            ovx, ovy = o.velocity
            t = ((ox - ax) * dx + (oy - ay) * dy) * inv_len_sq
            if t <= 0.02 or t > 1.02:
                continue

            tc = min(t, 1.0)
            distance = tc * length
            if self.fric == 1.0:
                flight_t = distance / ball_speed
            else:
                remainder = (
                    1.0
                    - distance * (1.0 - self.fric)
                    / (self.dt * ball_speed)
                )
                if remainder <= 0.0:
                    return -10.0
                flight_t = self.dt * _log(remainder) / self._log_fric

            pox = ox + ovx * flight_t * 0.5
            poy = oy + ovy * flight_t * 0.5
            cx, cy = ax + dx * tc, ay + dy * tc
            dist = _hypot(pox - cx, poy - cy) - self.body_radius
            req = SHOT_MARGIN + 0.025 * (tc * length)
            margin = dist - req

            if ox > self.half_w - 12.0 and abs(oy) < self.gw + 4.0:
                lateral = _hypot(cx - ox, cy - oy)
                toward = (
                    ovx * (cx - ox) + ovy * (cy - oy)
                ) / max(lateral, 1e-6)
                toward = clamp(toward, -self.vmax, self.vmax)
                reach_time = (
                    max(0.0, lateral - self.body_radius) / self.vmax
                )
                reach_time += (
                    0.45 * (self.vmax - toward) / self.acc
                )
                margin = min(
                    margin,
                    self.vmax * (reach_time + 0.05 - flight_t),
                )

            if margin < best_margin:
                best_margin = margin

        return best_margin

    def _self_blocked(self, p, bx, by, dx, dy):
        length = _hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        px, py = p.position[0] - bx, p.position[1] - by
        return (
            px * ux + py * uy > 0.0
            and abs(px * uy - py * ux) < self.body_radius
        )

    def _on_ball(self, obs, p, opps, opp_xy, keeper):
        ball = obs.ball
        bx, by = ball.position
        bvx, bvy = ball.velocity
        W2 = self.half_w
        imp = self.impulse
        near_sq = 1e+18

        for ox, oy in opp_xy:
            d_sq = (ox - bx) ** 2 + (oy - by) ** 2
            if d_sq < near_sq:
                near_sq = d_sq

        pressed = near_sq < PRESSURE_RADIUS_SQ
        gd = _hypot(W2 - bx, by)

        if not keeper and gd < SHOT_RANGE and bx > 2.0:
            best_m, best_y = -1000000000.0, 0.0
            for ty in self.shot_ys:
                dx_s, dy_s = W2 - bx, ty - by
                if self._self_blocked(p, bx, by, dx_s, dy_s):
                    continue
                m = self._shot_clearance(
                    bx, by, W2, ty, opps, bvx, bvy
                )
                corner_bonus = 0.15 if abs(ty) > 5.0 else 0.0
                if m + corner_bonus > best_m:
                    best_m, best_y = m + corner_bonus, ty

            if best_m >= 1.0 or (bx > W2 - 7 and best_m > -0.4):
                dx, dy = W2 - bx, best_y - by
                return self._strike(bvx, bvy, dx, dy, 30.0), False

        best = self._find_pass(obs, p, opps, pressed, keeper)
        if best is not None:
            tx, ty, kd, pw, receiver = best
            dx, dy = tx - bx, ty - by
            length = _hypot(dx, dy) or 1.0
            return _action((dx / length, dy / length), kd, pw), True

        if keeper or (pressed and bx < -18.0):
            bestc, best_vec = -1000000000.0, None
            for dx_c, ty_c in (
                (30.0, -24.0),
                (30.0, 24.0),
                (22.0, -self.half_h + 2.0),
                (22.0, self.half_h - 2.0),
                (16.0, -18.0),
                (16.0, 18.0),
            ):
                dy_c = ty_c - by
                if self._self_blocked(p, bx, by, dx_c, dy_c):
                    continue
                c = self.lane_clearance(
                    bx, by,
                    clamp(
                        bx + dx_c,
                        -self.half_w + 2.0,
                        self.half_w - 2.0,
                    ),
                    ty_c,
                    opp_xy,
                )
                if c > bestc:
                    bestc, best_vec = c, (dx_c, dy_c)

            if best_vec is not None:
                return self._strike(
                    bvx, bvy, best_vec[0], best_vec[1], 0.88 * imp
                ), False

        aim_x = W2 if bx > 28.0 else 38.0
        aim_y = clamp(by * 0.4, -5.0, 5.0)
        if bx > 10.0:
            aim_x = min(W2 - 5.0, max(bx + 5.0, W2 - 13.0))
            aim_y = 5.0 if by >= 0 else -5.0

        gdx, gdy = aim_x - bx, aim_y - by
        gn = _hypot(gdx, gdy) or 1.0
        gdx, gdy = gdx / gn, gdy / gn
        roll = self.dribble_roll
        best_action = None
        best_value = -1e+18
        acc = self.acc
        vmax = self.vmax
        dt = self.dt
        friction = self.fric
        player_pos = p.position
        player_vel = p.velocity

        # A candidate ends 4.5 units away. Beyond 13 units, an
        # opponent cannot lower either capped clearance (8 and 5).
        relative_opps = []
        for ox, oy in opp_xy:
            rx, ry = ox - bx, oy - by
            distance_sq = rx * rx + ry * ry
            if distance_sq < 169.0:
                relative_opps.append((rx, ry, distance_sq))
        moving_opps = []
        # Even a maximal impulse plus both players' motion cannot bring
        # a more distant opponent within the 2.5-unit danger threshold.
        danger_radius_sq = (
            (30.0 + self.impulse + self.vmax) * self.dt * 5.0 + 3.0
        ) ** 2
        for opponent in opps:
            ox, oy = opponent.position
            if (ox - bx) ** 2 + (oy - by) ** 2 > danger_radius_sq:
                continue
            ovx, ovy = opponent.velocity
            moving_opps.append((
                bx - ox,
                by - oy,
                ox + ovx * dt * 5,
                oy + ovy * dt * 5,
                min(
                    1.0,
                    getattr(opponent, 'kick_cooldown_ticks', 0) / 5.0,
                ),
            ))

        impulse = self.impulse
        for ang, ca, sa, ceiling in self.dribble_angles:
            if ceiling < best_value - 1e-9:
                break

            hx, hy = gdx * ca - gdy * sa, gdx * sa + gdy * ca
            nearest_sq = 64.0
            lane_sq = 25.0
            qdx, qdy = hx * 4.5, hy * 4.5

            for rx, ry, origin_sq in relative_opps:
                dx, dy = rx - qdx, ry - qdy
                end_sq = dx * dx + dy * dy
                if end_sq < nearest_sq:
                    nearest_sq = end_sq

                projection = rx * hx + ry * hy
                if projection < 0.0:
                    distance_sq = origin_sq
                elif projection > 4.5:
                    distance_sq = end_sq
                else:
                    cross = rx * hy - ry * hx
                    distance_sq = cross * cross

                if distance_sq < lane_sq:
                    lane_sq = distance_sq

            nearest = _sqrt(nearest_sq)
            lane = _sqrt(lane_sq)
            value = (
                nearest * 0.65 + lane * 1.3
                + 5.5 * ca - 0.35 * abs(ang)
            )

            if value <= best_value:
                continue

            px, py = player_pos
            vx, vy = player_vel
            future = []
            for _ in range(5):
                vx = vx * 0.9 + hx * acc * dt
                vy = vy * 0.9 + hy * acc * dt
                if vx * vx + vy * vy > vmax * vmax:
                    vn = _hypot(vx, vy)
                    vx *= vmax / vn
                    vy *= vmax / vn
                px += vx * dt
                py += vy * dt
                future.append((px, py))

            tx, ty = px + hx * 1.55, py + hy * 1.55
            if (
                abs(ty) > self.half_h - 1.0
                or tx > W2 - 0.6
                or tx < self.goal_x + 1.0
            ):
                continue

            desired_vx = (tx - bx) / roll
            desired_vy = (ty - by) / roll
            speed = _hypot(desired_vx, desired_vy)
            if speed < 0.1:
                continue

            kd, pw = self._strike_vec(
                bvx, bvy,
                desired_vx / speed, desired_vy / speed,
                speed,
            )
            actual_vx = bvx + kd[0] * pw * impulse
            actual_vy = bvy + kd[1] * pw * impulse
            x, y = bx, by
            minimum_gap = 1e+18
            maximum_gap = 0.0

            for px, py in future:
                x += actual_vx * dt
                y += actual_vy * dt
                actual_vx *= friction
                actual_vy *= friction
                gap = _hypot(x - px, y - py)
                if gap < minimum_gap:
                    minimum_gap = gap
                if gap > maximum_gap:
                    maximum_gap = gap

            gap_penalty = max(0.0, 1.47 - minimum_gap) * 30.0
            reach_penalty = max(0.0, maximum_gap - 2.2) * 12.0
            blocked = self._self_blocked(
                p, bx, by, desired_vx, desired_vy
            )
            fixed_penalties = (
                gap_penalty + reach_penalty
                + (20.0 if blocked else 0.0)
            )

            if value - fixed_penalties < best_value - 1e-9:
                continue

            danger = 12.0
            reject = False
            for dx0, dy0, ox1, oy1, cooldown in moving_opps:
                dx1, dy1 = x - ox1, y - oy1
                if (
                    (dx0 > 2.5 and dx1 > 2.5)
                    or (dx0 < -2.5 and dx1 < -2.5)
                ):
                    continue
                if (
                    (dy0 > 2.5 and dy1 > 2.5)
                    or (dy0 < -2.5 and dy1 < -2.5)
                ):
                    continue

                sx, sy = dx1 - dx0, dy1 - dy0
                denom = sx * sx + sy * sy
                t = (
                    clamp(
                        -(dx0 * sx + dy0 * sy) / denom,
                        0.0, 1.0,
                    )
                    if denom > 1e-12
                    else 0.0
                )
                distance = _hypot(dx0 + t * sx, dy0 + t * sy)
                if t < cooldown:
                    distance = min(
                        distance + 0.9,
                        _hypot(
                            dx0 + cooldown * sx,
                            dy0 + cooldown * sy,
                        ),
                    )
                danger = min(danger, distance)

                if (
                    value - fixed_penalties
                    - max(0.0, 2.5 - danger) * 20.0
                    < best_value - 1e-9
                ):
                    reject = True
                    break

            if reject:
                continue

            value -= max(0.0, 2.5 - danger) * 20.0
            value -= gap_penalty
            value -= reach_penalty
            if blocked:
                value -= 20.0

            if value > best_value:
                best_value = value
                best_action = _action(
                    movement=(hx, hy),
                    kick_direction=kd,
                    kick_power=pw,
                )

        if best_action is not None:
            return best_action, False
        return _action(movement=(gdx, gdy)), False

    def _strike(self, bvx, bvy, dx, dy, speed):
        n = _hypot(dx, dy) or 1.0
        ux, uy = dx / n, dy / n
        kd, pw = self._strike_vec(bvx, bvy, ux, uy, speed)
        return _action(
            movement=(ux, uy),
            kick_direction=kd,
            kick_power=pw,
        )

    def _strike_vec(self, bvx, bvy, ux, uy, speed):
        along = bvx * ux + bvy * uy
        transverse = bvx * uy - bvy * ux
        remaining = self.impulse * self.impulse - transverse * transverse
        if remaining >= 0.0:
            root = _sqrt(remaining)
            speed = min(along + root, max(along - root, speed))

        wx, wy = ux * speed - bvx, uy * speed - bvy
        m = _hypot(wx, wy)
        if m < 1e-06:
            return (ux, uy), 0.0
        return (wx / m, wy / m), min(m / self.impulse, 1.0)

    def _hybrid_defense(
        self, defenders, opps, bx, by, bvx, bvy, ours,
        carrier_id=None,
    ):
        if not defenders:
            return {}

        gx = self.goal_x
        pred_bx = bx + bvx * self.dt * 4.0
        pred_by = by + bvy * self.dt * 4.0

        if ours and bx > 0.0:
            screen_x = clamp(bx - SCREEN_DIST_OURS, gx + 10.0, 12.0)
        else:
            screen_x = clamp(pred_bx - SCREEN_DIST, gx + 4.5, -6.0)

        blend = clamp(
            (screen_x - gx) / max(3.0, pred_bx - gx),
            0.0, 1.0,
        )
        screen_y = clamp(
            pred_by * blend,
            -self.half_h + 3.0,
            self.half_h - 3.0,
        )
        screen_spot = screen_x, screen_y

        if len(defenders) == 1:
            return {defenders[0].id: screen_spot}

        d0, d1 = defenders[0], defenders[1]
        dist0 = (
            (d0.position[0] - screen_spot[0]) ** 2
            + (d0.position[1] - screen_spot[1]) ** 2
        )
        dist1 = (
            (d1.position[0] - screen_spot[0]) ** 2
            + (d1.position[1] - screen_spot[1]) ** 2
        )
        screener, marker = (d0, d1) if dist0 <= dist1 else (d1, d0)
        out = {screener.id: screen_spot}

        if carrier_id is None:
            carrier_id = (
                min(
                    opps,
                    key=lambda o: (
                        (o.position[0] - bx) ** 2
                        + (o.position[1] - by) ** 2
                    ),
                ).id
                if opps
                else -1
            )

        threats = [
            o for o in opps
            if o.id != carrier_id and o.position[0] < 18.0
        ]
        if threats:
            target_opp = min(threats, key=lambda o: o.position[0])
            ox, oy = target_opp.position
            dx, dy = gx - ox, -oy
            d = _hypot(dx, dy) or 1.0
            mx = clamp(
                ox + dx / d * MARK_GAP,
                gx + 4.5,
                min(bx - 4.0, 5.0),
            )
            my = clamp(
                oy + dy / d * MARK_GAP,
                -self.half_h + 3.0,
                self.half_h - 3.0,
            )
            out[marker.id] = mx, my
        else:
            offset_y = 5.5 if screen_y <= 0.0 else -5.5
            out[marker.id] = (
                clamp(screen_x - 2.0, gx + 4.5, -6.0),
                clamp(
                    screen_y + offset_y,
                    -self.half_h + 3.0,
                    self.half_h - 3.0,
                ),
            )

        for extra in defenders[2:]:
            out[extra.id] = screen_x, 0.0

        return out

    def _space_target(self, p, opp_xy, bx, by, ours, taken, tick):
        pid = p.id
        prev = self.targets.get(pid)
        if prev is not None and (tick + pid) % 5 != 0:
            return prev

        px, py = p.position
        anchor_x = bx if ours else clamp(bx + 12.0, -8.0, 20.0)
        xmax = self.half_w - 4.5
        best = prev if prev is not None else (px, py)
        best_s = -1000000000.0
        last_x = None
        max_clearance_sq = (8.0 + self.rad) ** 2

        for dx in (10.0, 22.0):
            x = clamp(anchor_x + dx, -6.0, xmax)
            if x == last_x:
                continue
            last_x = x
            lx = x - bx

            for y in (
                (-10.0, 0.0, 10.0)
                if x > 24.0
                else (-18.0, 0.0, 18.0)
            ):
                od_sq = 196.0
                lane_sq = max_clearance_sq
                ly = y - by
                length_sq = lx * lx + ly * ly
                inv_len_sq = (
                    1.0 / length_sq if length_sq > 1e-12 else 0.0
                )

                for ox, oy in opp_xy:
                    d_sq = (ox - x) ** 2 + (oy - y) ** 2
                    if d_sq < od_sq:
                        od_sq = d_sq

                    if ours and inv_len_sq:
                        t = (
                            (ox - bx) * lx + (oy - by) * ly
                        ) * inv_len_sq
                        if t < 0.0:
                            continue
                        if t > 1.0:
                            t = 1.0
                        cx, cy = bx + lx * t, by + ly * t
                        d_sq = (ox - cx) ** 2 + (oy - cy) ** 2
                        if d_sq < lane_sq:
                            lane_sq = d_sq

                od = _sqrt(od_sq)
                box_bonus = (
                    -0.12 * abs(y) if x > 24.0 else 0.05 * abs(y)
                )
                s = (
                    od + 0.25 * x + box_bonus
                    - 0.09 * _hypot(x - px, y - py)
                )

                if ours:
                    s += (
                        0.65 * (_sqrt(lane_sq) - self.rad)
                        if inv_len_sq
                        else 0.0
                    )

                for tx, ty in taken:
                    td = _hypot(x - tx, y - ty)
                    if td < 12.0:
                        s -= (12.0 - td) * 1.1

                if (
                    prev is not None
                    and (x - prev[0]) ** 2 + (y - prev[1]) ** 2 < 9.0
                ):
                    s += 1.8

                if s > best_s:
                    best_s, best = s, (x, y)

        self.targets[pid] = best
        return best

    def _find_pass(self, obs, p, opps, pressed, keeper):
        bx, by = obs.ball.position
        bvx, bvy = obs.ball.velocity
        best = None
        best_score = -1e9
        opponents = [
            (
                o.position,
                o.velocity,
                getattr(o, 'kick_cooldown_ticks', 0) * self.dt,
            )
            for o in opps
        ]
        drag = self._pass_drag
        goal_distance = _hypot(self.half_w - bx, by)
        ppx, ppy = p.position
        ppx -= bx
        ppy -= by

        for mate in obs.my_players:
            if mate.id in (0, p.id):
                continue

            mx, my = mate.position
            vx, vy = mate.velocity
            lead_time = min(1.5, _hypot(mx - bx, my - by) / 20.0)
            choices = (
                (mx, my),
                (mx + vx * lead_time, my + vy * lead_time),
                (mx + 6.0, my),
                (mx + 16, my),
                (mx + 6, my - 6),
                (mx + 6, my + 6),
            )

            for tx, ty in choices:
                tx = clamp(
                    tx, -self.half_w + 2.0, self.half_w - 2.0
                )
                ty = clamp(
                    ty, -self.half_h + 2.0, self.half_h - 2.0
                )
                dx, dy = tx - bx, ty - by
                length = _hypot(dx, dy)
                if length < 3 or length > 45 or tx < bx - 8:
                    continue

                ux, uy = dx / length, dy / length
                if (
                    ppx * ux + ppy * uy > 0.0
                    and abs(ppx * uy - ppy * ux) < self.body_radius
                ):
                    continue

                gain = tx - bx
                goal_improvement = (
                    goal_distance - _hypot(self.half_w - tx, ty)
                )
                ceiling = 0.55 * gain + 0.3 * goal_improvement + 4.5

                if ceiling <= best_score:
                    continue
                if not pressed and not keeper and ceiling < 5.0:
                    continue

                desired = 30.0
                kd, pw = self._strike_vec(
                    bvx, bvy, ux, uy, desired
                )
                ax = bvx + kd[0] * pw * self.impulse
                ay = bvy + kd[1] * pw * self.impulse
                actual = _hypot(ax, ay)

                if actual < 5.0 or abs(ax * uy - ay * ux) > 0.1:
                    continue
                actual = min(30.0, actual)
                remainder = 1.0 - length * drag / actual
                if remainder <= 0.05:
                    continue

                flight = self.dt * _log(remainder) / self._log_fric
                receiver_distance = _hypot(tx - mx, ty - my)
                rv = (
                    vx * (tx - mx) + vy * (ty - my)
                ) / max(receiver_distance, 1e-6)
                receive_time = (
                    max(0, receiver_distance - self.krange)
                    / self.vmax
                )
                receive_time += (
                    0.45
                    * (self.vmax - clamp(rv, -self.vmax, self.vmax))
                    / self.acc
                    if receiver_distance > self.krange
                    else 0.0
                )
                receive_time = max(
                    receive_time,
                    getattr(mate, 'kick_cooldown_ticks', 0) * self.dt,
                )
                if receive_time > flight + 0.10:
                    continue

                upper = (
                    0.55 * gain + 0.3 * goal_improvement
                    + 4.5 - flight
                )
                if upper <= best_score:
                    continue
                if not pressed and not keeper and upper < 5.0:
                    continue

                required_margin = (
                    0.05 if abs(dy) > 5 and tx > bx - 3 else 0.15
                )
                margin = 10.0

                for (ox, oy), (ovx, ovy), cooldown in opponents:
                    along = (ox - bx) * ux + (oy - by) * uy
                    if along < 0.0:
                        continue
                    along = min(length, along)

                    f = (
                        self.dt
                        * _log(max(
                            0.05,
                            1.0 - along * drag / actual,
                        ))
                        / self._log_fric
                    )
                    cx, cy = bx + ux * along, by + uy * along
                    od = _hypot(cx - ox, cy - oy)
                    toward = (
                        ovx * (cx - ox) + ovy * (cy - oy)
                    ) / max(od, 1e-6)

                    reach = max(0.0, od - self.krange) / self.vmax
                    reach += (
                        0.45
                        * (
                            self.vmax
                            - clamp(toward, -self.vmax, self.vmax)
                        )
                        / self.acc
                        if od > self.krange
                        else 0.0
                    )

                    if cooldown > reach:
                        body_reach = (
                            max(0.0, od - self.body_radius)
                            / self.vmax
                        )
                        if od > self.body_radius:
                            body_reach += (
                                0.45
                                * (
                                    self.vmax
                                    - clamp(
                                        toward,
                                        -self.vmax,
                                        self.vmax,
                                    )
                                )
                                / self.acc
                            )
                        reach = min(body_reach, cooldown)

                    margin = min(margin, reach - f)
                    if margin < required_margin:
                        break

                if margin < required_margin:
                    continue

                value = (
                    0.55 * gain
                    + 0.3 * goal_improvement
                    + min(margin, 1.5) * 3.0
                    - flight
                )
                if not pressed and not keeper and value < 5.0:
                    continue

                if value > best_score:
                    best_score = value
                    best = tx, ty, kd, pw, mate.id

        return best