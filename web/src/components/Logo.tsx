/**
 * The TailSafe mark: a tower in section, its floors ruled, with the long
 * right-hand tail of a distribution drawn down its side — the part of the
 * outcome the tool is about.
 */
export function Logo({ size = 30, tail = "var(--go)" }: { size?: number; tail?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden focusable="false">
      <rect x="6" y="3.5" width="11" height="25" fill="none" stroke="currentColor" strokeWidth="2" />
      {[8, 12, 16, 20, 24].map((y) => (
        <line key={y} x1="8.2" x2="14.8" y1={y} y2={y} stroke="currentColor" strokeWidth="1.2" opacity="0.6" />
      ))}
      <path
        d="M17 5.5c1.2 0 2 3.5 3 8.5s2.8 9.6 5.2 11.9c.9.9 2 1.4 3.3 1.6"
        fill="none"
        stroke={tail}
        strokeWidth="2.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
