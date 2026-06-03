import { createRoot } from "solid-js";

import { __resetPersistenceForTests, enablePersistence } from "../persist";
import { initialClockMode, initialThemeMode, setClockMode, setThemeMode, setViewMode } from "../signals";

describe("persist", () => {
  beforeEach(() => {
    __resetPersistenceForTests();
    setClockMode("wall");
    setViewMode("layered");
    setThemeMode("dark");
    window.localStorage.clear();
    document.documentElement.classList.remove("theme-light");
  });

  it("writes the view mode into localStorage", async () => {
    createRoot((dispose) => {
      enablePersistence();
      setViewMode("concise");
      queueMicrotask(dispose);
    });

    await Promise.resolve();
    expect(window.localStorage.getItem("gantt.viewMode")).toBe("concise");
  });

  it("writes the clock mode into localStorage", async () => {
    createRoot((dispose) => {
      enablePersistence();
      setClockMode("real");
      queueMicrotask(dispose);
    });

    await Promise.resolve();
    expect(window.localStorage.getItem("gantt.clockMode")).toBe("real");
  });

  it("writes the theme mode into localStorage and toggles the root class", async () => {
    createRoot((dispose) => {
      enablePersistence();
      setThemeMode("light");
      queueMicrotask(dispose);
    });

    await Promise.resolve();
    expect(window.localStorage.getItem("gantt.themeMode")).toBe("light");
    expect(document.documentElement.classList.contains("theme-light")).toBe(true);
  });

  it("restores light mode from localStorage", () => {
    window.localStorage.setItem("gantt.themeMode", "light");
    expect(initialThemeMode()).toBe("light");
  });

  it("restores dark mode from localStorage", () => {
    window.localStorage.setItem("gantt.themeMode", "dark");
    expect(initialThemeMode()).toBe("dark");
  });

  it("restores real mode from localStorage", () => {
    window.localStorage.setItem("gantt.clockMode", "real");
    expect(initialClockMode()).toBe("real");
  });
});
