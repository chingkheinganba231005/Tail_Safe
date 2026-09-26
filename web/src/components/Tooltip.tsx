import { useCallback, useState, type ReactNode } from "react";

export interface TooltipContent {
  title?: string;
  rows: { label: string; value: string; swatch?: string }[];
  note?: string;
}

interface State {
  x: number;
  y: number;
  content: TooltipContent;
}

/**
 * Hover tooltip: values lead, labels follow, swatch carries identity. React
 * renders text as text (never HTML), so data can't inject markup.
 */
export function useTooltip() {
  const [state, setState] = useState<State | null>(null);
  const show = useCallback((e: { clientX: number; clientY: number }, content: TooltipContent) => {
    setState({ x: e.clientX, y: e.clientY, content });
  }, []);
  const hide = useCallback(() => setState(null), []);
  let node: ReactNode = null;
  if (state) {
    const flip = state.x > window.innerWidth - 260;
    node = (
      <div
        className="tooltip"
        role="status"
        style={{
          left: flip ? undefined : state.x + 14,
          right: flip ? window.innerWidth - state.x + 14 : undefined,
          top: state.y + 14,
        }}
      >
        {state.content.title && (
          <div className="secondary" style={{ marginBottom: 2 }}>
            {state.content.title}
          </div>
        )}
        {state.content.rows.map((r) => (
          <div key={r.label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            {r.swatch && (
              <span
                aria-hidden
                style={{ width: 8, height: 8, borderRadius: 2, background: r.swatch }}
              />
            )}
            <strong className="tabular">{r.value}</strong>
            <span className="secondary">{r.label}</span>
          </div>
        ))}
        {state.content.note && <div className="muted">{state.content.note}</div>}
      </div>
    );
  }
  return { show, hide, node };
}
