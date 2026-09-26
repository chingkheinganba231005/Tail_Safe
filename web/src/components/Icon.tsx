import type { ReactNode, SVGProps } from "react";

/** A small set of line icons drawn on a 20-unit grid (1.6 stroke, round caps). */
const PATHS: Record<string, ReactNode> = {
  check: <path d="M4.5 10.5l3.5 3.5 7.5-8" />,
  alert: (
    <>
      <path d="M10 3.2l7.3 12.6a1 1 0 0 1-.9 1.5H3.6a1 1 0 0 1-.9-1.5L10 3.2z" />
      <path d="M10 8v3.8" />
      <circle cx="10" cy="14.4" r="0.35" fill="currentColor" />
    </>
  ),
  stop: (
    <>
      <path d="M7 2.8h6L17.2 7v6L13 17.2H7L2.8 13V7z" />
      <path d="M7.5 7.5l5 5M12.5 7.5l-5 5" />
    </>
  ),
  info: (
    <>
      <circle cx="10" cy="10" r="7.2" />
      <path d="M10 9v4.5" />
      <circle cx="10" cy="6.4" r="0.35" fill="currentColor" />
    </>
  ),
  arrow: <path d="M4 10h11.5M11 5.5l4.5 4.5-4.5 4.5" />,
  sun: (
    <>
      <circle cx="10" cy="10" r="3.2" />
      <path d="M10 2.5v1.8M10 15.7v1.8M2.5 10h1.8M15.7 10h1.8M4.7 4.7l1.3 1.3M14 14l1.3 1.3M4.7 15.3L6 14M14 6l1.3-1.3" />
    </>
  ),
  moon: <path d="M15.8 12.6A6.6 6.6 0 0 1 7.4 4.2a6.6 6.6 0 1 0 8.4 8.4z" />,
  auto: (
    <>
      <circle cx="10" cy="10" r="7" />
      <path d="M10 3v14a7 7 0 0 0 0-14z" fill="currentColor" stroke="none" />
    </>
  ),
  download: <path d="M10 3.5v9M6 8.8l4 4 4-4M4 16.5h12" />,
  lock: (
    <>
      <rect x="4.5" y="9" width="11" height="8" rx="1.5" />
      <path d="M7 9V6.8a3 3 0 0 1 6 0V9" />
    </>
  ),
  external: <path d="M8 4.5H5a1 1 0 0 0-1 1V15a1 1 0 0 0 1 1h9.5a1 1 0 0 0 1-1v-3M11.5 4h4.5v4.5M16 4l-7 7" />,
  play: <path d="M6.5 4.5v11l9-5.5z" />,
  pause: <path d="M7 4.5v11M13 4.5v11" />,
};

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, ...rest }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      focusable="false"
      {...rest}
    >
      {PATHS[name]}
    </svg>
  );
}

type Tone = "good" | "warning" | "critical" | "info";
const TONE: Record<Tone, { icon: IconName; color: string }> = {
  good: { icon: "check", color: "var(--good)" },
  warning: { icon: "alert", color: "var(--warning)" },
  critical: { icon: "stop", color: "var(--critical)" },
  info: { icon: "info", color: "var(--text-primary)" },
};

/** A status line: always an icon and words, never colour alone. */
export function Callout({
  tone,
  children,
  role,
  className = "",
}: {
  tone: Tone;
  children: ReactNode;
  role?: "alert" | "status" | "note";
  className?: string;
}) {
  const t = TONE[tone];
  return (
    <div
      className={`callout ${className}`}
      style={{ borderLeftColor: t.color }}
      role={role ?? (tone === "critical" ? "alert" : "status")}
    >
      <Icon name={t.icon} style={{ color: t.color }} />
      <div className="min-w-0">{children}</div>
    </div>
  );
}
