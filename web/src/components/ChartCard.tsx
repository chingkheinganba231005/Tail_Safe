import { useState, type ReactNode } from "react";

export interface TableData {
  columns: string[];
  rows: (string | number)[][];
}

interface Props {
  title: string;
  subtitle?: ReactNode;
  table?: TableData;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** A titled chart with a chart/table toggle (every chart has a table view). */
export function ChartCard({ title, subtitle, table, actions, children, className }: Props) {
  const [view, setView] = useState<"chart" | "table">("chart");
  return (
    <section className={`card p-4 sm:p-5 ${className ?? ""}`}>
      <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="text-[0.95rem] font-semibold">{title}</h3>
          {subtitle && <p className="secondary mt-0.5 max-w-prose text-sm leading-snug">{subtitle}</p>}
        </div>
        <div className="flex items-center gap-2">
          {actions}
          {table && (
            <div role="group" aria-label="View" className="segmented segmented-sm">
              {(["chart", "table"] as const).map((v) => (
                <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>
                  {v === "chart" ? "Chart" : "Table"}
                </button>
              ))}
            </div>
          )}
        </div>
      </header>
      {view === "chart" || !table ? children : <DataTable table={table} />}
    </section>
  );
}

export function DataTable({ table }: { table: TableData }) {
  return (
    <div className="max-h-80 overflow-auto rounded-md" style={{ border: "1px solid var(--border)" }}>
      <table className="tabular w-full text-sm">
        <thead>
          <tr>
            {table.columns.map((c, i) => (
              <th
                key={c}
                className={`sticky top-0 px-3 py-1.5 ${i ? "text-right" : "text-left"}`}
                style={{ background: "var(--surface-2)" }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((r, i) => (
            <tr key={i} style={{ borderTop: "1px solid var(--grid)" }}>
              {r.map((v, j) => (
                <td key={j} className={`px-3 py-1.5 ${j ? "text-right" : ""}`}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
