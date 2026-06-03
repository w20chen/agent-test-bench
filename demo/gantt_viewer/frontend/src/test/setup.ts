import { afterEach, vi } from "vitest";

class ResizeObserverStub {
  observe(): void {}

  disconnect(): void {}

  unobserve(): void {}
}

if (!("ResizeObserver" in globalThis)) {
  Object.defineProperty(globalThis, "ResizeObserver", {
    value: ResizeObserverStub,
    writable: true,
  });
}

Object.defineProperty(globalThis, "requestAnimationFrame", {
  value: (callback: FrameRequestCallback) => {
    callback(0);
    return 1;
  },
  writable: true,
});

Object.defineProperty(globalThis, "cancelAnimationFrame", {
  value: () => {},
  writable: true,
});

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});
