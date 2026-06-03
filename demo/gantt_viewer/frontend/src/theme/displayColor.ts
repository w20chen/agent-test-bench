const LIGHT_THEME_COLOR_MAP: Record<string, string> = {
  "#00e5ff": "#0b6e95",
  "#ff6d00": "#b05a10",
  "#76ff03": "#4d8f12",
  "#ab47bc": "#8b4fa8",
  "#ff1744": "#b91c1c",
  "#6b7280": "#64748b",
  "#94a3b8": "#64748b",
};

export function isLightThemeActive(): boolean {
  return document.documentElement.classList.contains("theme-light");
}

export function displayColor(rawColor: string): string {
  if (!isLightThemeActive()) {
    return rawColor;
  }
  return LIGHT_THEME_COLOR_MAP[rawColor.toLowerCase()] ?? rawColor;
}
