import { useEffect, useState } from "react";
import type { Mode } from "./color";

/** Current colour mode: an explicit data-theme wins, else the OS preference. */
export function currentMode(): Mode {
  if (typeof document === "undefined") return "light";
  const forced = document.documentElement.dataset.theme;
  if (forced === "light" || forced === "dark") return forced;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function useMode(): Mode {
  const [mode, setMode] = useState<Mode>(currentMode);
  useEffect(() => {
    const update = () => setMode(currentMode());
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    mq?.addEventListener("change", update);
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq?.removeEventListener("change", update);
      obs.disconnect();
    };
  }, []);
  return mode;
}

/** Read a CSS custom property (for canvas / WebGL, which cannot use var()). */
export function cssVar(name: string): string {
  if (typeof document === "undefined") return "#888888";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888888";
}
