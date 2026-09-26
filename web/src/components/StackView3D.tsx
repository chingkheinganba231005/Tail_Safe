import { Edges, OrbitControls } from "@react-three/drei";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import { CanvasTexture, DoubleSide, Shape, ShapeGeometry, Vector2, type BufferGeometry } from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { BLUE, ORANGE, ramp, smokeLevel } from "../lib/color";
import { levelLabel } from "../lib/format";
import { useMode } from "../lib/theme";
import type { Building, Replay } from "../types";
import { useTooltip } from "./Tooltip";

interface Props {
  building: Building;
  replay?: Replay | null;
  frame?: number;
  highlightEdges?: Set<string>;
  highlightStair?: string | null;
  queueMax?: number;
  height?: number;
}

interface LevelGeo {
  index: number;
  y: number; // elevation (three.js y is up)
  h: number;
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
  footprint: BufferGeometry | null; // merged room polygons, horizontal at y = 0
}

/** A text label drawn on a canvas sprite (no DOM overlay, so nothing to clean up). */
function Label({ text, position, color, size }: { text: string; position: [number, number, number]; color: string; size: number }) {
  const texture = useMemo(() => {
    const c = document.createElement("canvas");
    c.width = 160;
    c.height = 48;
    const ctx = c.getContext("2d");
    if (ctx) {
      ctx.font = "28px system-ui, sans-serif";
      ctx.fillStyle = color;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 156, 24);
    }
    return new CanvasTexture(c);
  }, [text, color]);
  useEffect(() => () => texture.dispose(), [texture]);
  return (
    <sprite position={position} scale={[size * 3.33, size, 1]}>
      <spriteMaterial map={texture} transparent depthWrite={false} />
    </sprite>
  );
}

/** Rooms of one level merged into one flat geometry (plan y → -z). */
function footprint(building: Building, level: number): BufferGeometry | null {
  const parts = building.nodes
    .filter((n) => n.level === level && n.type !== "exit" && n.polygon && n.polygon.length >= 3)
    .map((n) => {
      const shape = new Shape(n.polygon!.map(([x, y]) => new Vector2(x, y)));
      const g = new ShapeGeometry(shape);
      g.rotateX(-Math.PI / 2);
      return g;
    });
  if (parts.length === 0) return null;
  const merged = mergeGeometries(parts, false);
  parts.forEach((g) => g.dispose());
  return merged;
}

/**
 * The tower as stacked translucent floors. Floor colour: smoke (orange ramp,
 * from corridor visibility). Stair columns: queue length on each flight (blue
 * ramp). Hover a floor for its numbers.
 */
export function StackView3D({
  building,
  replay,
  frame = 0,
  highlightEdges,
  highlightStair,
  queueMax,
  height = 520,
}: Props) {
  const mode = useMode();
  const tip = useTooltip();
  const geo = useMemo(() => {
    const levels: LevelGeo[] = building.levels.map((lv) => {
      const pts = building.nodes
        .filter((n) => n.level === lv.index && n.type !== "exit")
        .flatMap((n) => n.polygon ?? [[n.x, n.y]]);
      const xs = pts.map((p) => p[0]);
      const zs = pts.map((p) => -p[1]);
      return {
        index: lv.index,
        y: lv.elevation,
        h: lv.height,
        minX: Math.min(...xs),
        maxX: Math.max(...xs),
        minZ: Math.min(...zs),
        maxZ: Math.max(...zs),
        footprint: footprint(building, lv.index),
      };
    });
    const stairs = building.stairs.map((s) => ({
      id: s.id,
      landings: building.nodes
        .filter((n) => n.type === "stair_landing" && n.stair === s.id)
        .map((n) => ({ level: n.level, x: n.x, z: -n.y })),
    }));
    const byId = new Map(building.nodes.map((n) => [n.id, n]));
    const top = Math.max(...levels.map((l) => l.y + l.h));
    const span = Math.max(...levels.map((l) => Math.max(l.maxX - l.minX, l.maxZ - l.minZ)));
    // Look across the line joining the first two stairs so they don't hide each other.
    let view: [number, number] = [0.62, 0.78];
    const a = stairs[0]?.landings.find((x) => x.level > 0);
    const b = stairs[1]?.landings.find((x) => x.level > 0);
    if (a && b) {
      const dx = b.x - a.x;
      const dz = b.z - a.z;
      const n = Math.hypot(dx, dz) || 1;
      view = [-dz / n, dx / n];
      if (view[0] + view[1] < 0) view = [-view[0], -view[1]];
      // Tilt 25° off the perpendicular so the plan reads as 3D.
      const c = Math.cos(0.44);
      const s = Math.sin(0.44);
      view = [view[0] * c - view[1] * s, view[0] * s + view[1] * c];
    }
    return { levels, stairs, byId, top, span, view };
  }, [building]);
  useEffect(() => () => geo.levels.forEach((l) => l.footprint?.dispose()), [geo]);

  const marks = useMemo(() => {
    if (!highlightEdges?.size) return [];
    return building.edges
      .filter((e) => highlightEdges.has(e.id))
      .map((e) => {
        const s = geo.byId.get(e.source)!;
        const t = geo.byId.get(e.target)!;
        const lv = building.levels.find((l) => l.index === Math.max(s.level, t.level));
        return {
          id: e.id,
          x: (s.x + t.x) / 2,
          y: (lv?.elevation ?? 0) + 0.8,
          z: -(s.y + t.y) / 2,
        };
      });
  }, [building, geo, highlightEdges]);

  const f = replay ? Math.min(frame, replay.times.length - 1) : 0;
  const levelPos = new Map(replay?.levels.map((lv, i) => [lv, i]) ?? []);
  const qMax =
    queueMax ??
    Math.max(
      5,
      ...(replay ? Object.values(replay.stair_queues).flatMap((g) => g.flatMap((r) => r)) : [0]),
    );
  const neutral = mode === "dark" ? "#8a8983" : "#c9c8c0";
  const blockedFill = mode === "dark" ? "#0d0d0d" : "#52514e";
  const ink = mode === "dark" ? "#ffffff" : "#0b0b0b";
  const muted = mode === "dark" ? "#c3c2b7" : "#52514e";
  const blocked = replay?.info.blocked_stairs ?? {};
  const now = replay ? replay.times[f] : 0;

  const hover = (e: ThreeEvent<PointerEvent>, level: number) => {
    e.stopPropagation();
    const k = levelPos.get(level);
    const rows = [];
    if (replay && k !== undefined) {
      rows.push({ label: "people still on the floor", value: String(replay.remaining[f][k]) });
      const vis = replay.visibility?.[f][k];
      if (vis !== undefined) rows.push({ label: "corridor visibility", value: `${vis.toFixed(0)} m` });
      for (const [sid, grid] of Object.entries(replay.stair_queues)) {
        rows.push({ label: `queued at Stair ${sid}`, value: grid[f][k].toFixed(0) });
      }
    }
    tip.show(e.nativeEvent, { title: levelLabel(level), rows });
  };

  const cx = (geo.levels[0].minX + geo.levels[0].maxX) / 2;
  const cz = (geo.levels[0].minZ + geo.levels[0].maxZ) / 2;
  const dist = Math.max(geo.top, geo.span) * 1.6;
  const [vx, vz] = geo.view;

  return (
    <div className="relative w-full" style={{ height }}>
      <Canvas
        camera={{ position: [cx + dist * vx, geo.top * 0.75, cz + dist * vz], fov: 40, near: 0.5, far: dist * 10 }}
        style={{ background: "var(--surface-1)", borderRadius: 8 }}
        onPointerMissed={tip.hide}
      >
        <ambientLight intensity={0.9} />
        <directionalLight position={[50, 200, 80]} intensity={0.6} />
        <OrbitControls target={[cx, geo.top * 0.45, cz]} enableDamping />
        {geo.levels.map((l) => {
          const k = levelPos.get(l.index);
          const smoke = replay && k !== undefined ? smokeLevel(replay.visibility?.[f][k]) : 0;
          const color = smoke > 0.02 ? ramp(ORANGE, smoke, mode) : neutral;
          const w = l.maxX - l.minX;
          const d = l.maxZ - l.minZ;
          return (
            <group key={l.index}>
              {l.footprint ? (
                <mesh
                  geometry={l.footprint}
                  position={[0, l.y + 0.05, 0]}
                  onPointerMove={(e) => hover(e, l.index)}
                  onPointerOut={tip.hide}
                >
                  <meshStandardMaterial
                    color={color}
                    transparent
                    opacity={smoke > 0.02 ? 0.55 + 0.35 * smoke : 0.3}
                    depthWrite={false}
                    side={DoubleSide}
                  />
                </mesh>
              ) : (
                <mesh position={[(l.minX + l.maxX) / 2, l.y + 0.15, (l.minZ + l.maxZ) / 2]}>
                  <boxGeometry args={[w, 0.3, d]} />
                  <meshStandardMaterial color={color} transparent opacity={0.3} depthWrite={false} />
                </mesh>
              )}
              {l.index % 5 === 0 && (
                <Label
                  text={levelLabel(l.index)}
                  position={[l.minX - 1, l.y + 0.5, l.maxZ]}
                  color={muted}
                  size={Math.max(2.4, geo.span / 12)}
                />
              )}
            </group>
          );
        })}
        {geo.stairs.map((s) =>
          s.landings.map((ld) => {
            const l = geo.levels.find((g) => g.index === ld.level);
            if (!l || ld.level === 0) return null;
            const k = levelPos.get(ld.level);
            const q = replay && k !== undefined ? replay.stair_queues[s.id]?.[f][k] ?? 0 : 0;
            const isBlocked = s.id in blocked && now >= blocked[s.id];
            const color = isBlocked ? blockedFill : q > 0.5 ? ramp(BLUE, q / qMax, mode) : neutral;
            const hl = highlightStair === s.id;
            return (
              <mesh
                key={`${s.id}-${ld.level}`}
                position={[ld.x, l.y - l.h / 2 + 0.3, ld.z]}
                onPointerMove={(e) => hover(e, ld.level)}
                onPointerOut={tip.hide}
              >
                <boxGeometry args={[2.2, l.h - 0.4, 2.2]} />
                <meshStandardMaterial color={color} transparent opacity={q > 0.5 || isBlocked ? 0.95 : 0.35} />
                {(hl || isBlocked) && <Edges color={hl ? ink : muted} lineWidth={hl ? 2 : 1} />}
              </mesh>
            );
          }),
        )}
        {marks.map((m) => (
          <mesh key={m.id} position={[m.x, m.y, m.z]}>
            <sphereGeometry args={[0.9, 16, 16]} />
            <meshStandardMaterial color={ink} />
          </mesh>
        ))}
      </Canvas>
      {tip.node}
    </div>
  );
}

/** Colour keys for the 3D view (continuous ramps with end labels). */
export function StackLegend({ mode, queueMax }: { mode: "light" | "dark"; queueMax: number }) {
  const grad = (stops: string[]) =>
    `linear-gradient(to right, ${Array.from({ length: 11 }, (_, i) => ramp(stops, i / 10, mode)).join(",")})`;
  return (
    <div className="secondary flex flex-wrap gap-x-8 gap-y-2 text-xs">
      <div>
        <div>Smoke on the floor (corridor visibility)</div>
        <div style={{ width: 160, height: 10, borderRadius: 3, background: grad(ORANGE) }} />
        <div className="muted flex justify-between tabular" style={{ width: 160 }}>
          <span>≥20 m</span>
          <span>2 m</span>
        </div>
      </div>
      <div>
        <div>People queued on the stair flight</div>
        <div style={{ width: 160, height: 10, borderRadius: 3, background: grad(BLUE) }} />
        <div className="muted flex justify-between tabular" style={{ width: 160 }}>
          <span>0</span>
          <span>{queueMax.toFixed(0)}+</span>
        </div>
      </div>
      <div className="flex items-end gap-2">
        <span
          style={{
            width: 12,
            height: 14,
            background: mode === "dark" ? "#0d0d0d" : "#52514e",
            border: "1px solid var(--text-secondary)",
            display: "inline-block",
          }}
        />
        <span>Stair blocked (smoke-logged)</span>
      </div>
    </div>
  );
}
