import { createEffect } from "solid-js";

import { clockMode, themeMode, viewMode } from "./signals";

let persistenceStarted = false;
let displaySyncStarted = false;

export function enableDisplaySync(): void {
  if (displaySyncStarted || typeof window === "undefined") {
    return;
  }
  displaySyncStarted = true;

  // Global CSS hooks for viewMode — lets .sidebar and canvas share lane
  // height through a single CSS variable instead of threading a prop.
  createEffect(() => {
    if (typeof document === "undefined") return;
    const concise = viewMode() === "concise";
    document.body.classList.toggle("view-concise", concise);
    document.documentElement.style.setProperty("--lane-h", concise ? "26px" : "60px");
  });

  createEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.classList.toggle("theme-light", themeMode() === "light");
  });
}

export function enablePersistence(): void {
  if (typeof window === "undefined") {
    return;
  }
  enableDisplaySync();
  if (persistenceStarted) {
    return;
  }
  persistenceStarted = true;

  createEffect(() => {
    window.localStorage.setItem("gantt.viewMode", viewMode());
  });

  createEffect(() => {
    window.localStorage.setItem("gantt.clockMode", clockMode());
  });

  createEffect(() => {
    window.localStorage.setItem("gantt.themeMode", themeMode());
  });
}

export function __resetPersistenceForTests(): void {
  persistenceStarted = false;
  displaySyncStarted = false;
}
