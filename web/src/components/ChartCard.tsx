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
    <section className={`card p-4 ${className ?? ""}`}>
      <header className="mb-2 flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold">{title}</h3>
          {subtitle && <p className="secondary text-sm">{subtitle}</p>}
        </div>
        <div className="flex items-center gap-2">
          {actions}
          {table && (
            <div role="group" aria-label="View" className="flex gap-1">
              {(["chart", "table"] as const).map((v) => (
                <button
                  key={v}
                  className="btn-ghost text-xs"
                  aria-pressed={view === v}
                  onClick={() => setView(v)}
                >
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
    <div className="max-h-80 overflow-auto">
      <table className="tabular w-full text-sm">
        <thead>
          <tr>
            {table.columns.map((c, i) => (
              <th
                key={c}
                className={`secondary sticky top-0 py-1 font-medium ${i ? "text-right" : "text-left"}`}
                style={{ background: "var(--surface-1)" }}
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
                <td key={j} className={`py-1 ${j ? "text-right" : ""}`}>
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
