/** Small plan drawings of the four building types (plan view, north up). */
export function TypologyGlyph({ name, size = 44 }: { name: string; size?: number }) {
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.4, strokeLinejoin: "round" as const };
  const core = { fill: "currentColor", opacity: 0.18, stroke: "none" };
  let body;
  switch (name) {
    case "cruciform":
      body = (
        <>
          <path d="M18 4h12v14h14v12H30v14H18V30H4V18h14z" {...common} />
          <rect x="19" y="19" width="10" height="10" {...core} />
          <path d="M24 6v10M24 32v10M6 24h10M32 24h10" stroke="currentColor" strokeWidth="0.8" opacity="0.6" />
        </>
      );
      break;
    case "slab":
      body = (
        <>
          <rect x="3" y="17" width="42" height="14" rx="0.5" {...common} />
          <path d="M5 24h38" stroke="currentColor" strokeWidth="0.8" opacity="0.6" />
          <rect x="3.7" y="17.7" width="4" height="12.6" {...core} />
          <rect x="40.3" y="17.7" width="4" height="12.6" {...core} />
          <rect x="21" y="17.7" width="6" height="5.6" {...core} />
          {[11, 16, 31, 36].map((x) => (
            <path key={x} d={`M${x} 17v5M${x} 26v5`} stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
          ))}
        </>
      );
      break;
    case "twin_core":
      body = (
        <>
          <rect x="4" y="30" width="40" height="14" rx="0.5" {...common} opacity={0.55} />
          <rect x="11" y="4" width="26" height="30" rx="0.5" {...common} />
          <rect x="19" y="13" width="10" height="12" {...core} />
          <path d="M20 15l8 8M28 15l-8 8" stroke="currentColor" strokeWidth="0.8" opacity="0.7" />
        </>
      );
      break;
    default:
      body = (
        <>
          <rect x="4" y="12" width="40" height="24" rx="0.5" {...common} />
          <path d="M6 24h36" stroke="currentColor" strokeWidth="0.8" opacity="0.6" />
          {[11, 18, 25, 32, 39].map((x) => (
            <path key={x} d={`M${x} 12v8M${x} 28v8`} stroke="currentColor" strokeWidth="0.8" opacity="0.5" />
          ))}
          <rect x="4.7" y="12.7" width="5" height="22.6" {...core} />
        </>
      );
  }
  return (
    <svg className="glyph" width={size} height={size} viewBox="0 0 48 48" aria-hidden focusable="false">
      {body}
    </svg>
  );
}
