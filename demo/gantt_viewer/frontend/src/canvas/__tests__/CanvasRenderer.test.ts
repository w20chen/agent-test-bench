import type { GanttPayload } from "../../api/client";
import { CanvasRenderer } from "../CanvasRenderer";

function createMockContext(): CanvasRenderingContext2D {
  const noop = () => {};
  return {
    arc: noop,
    beginPath: noop,
    closePath: noop,
    clearRect: noop,
    fill: noop,
    fillRect: noop,
    fillText: noop,
    lineTo: noop,
    moveTo: noop,
    restore: noop,
    rotate: noop,
    save: noop,
    setTransform: noop,
    stroke: noop,
    translate: noop,
  } as unknown as CanvasRenderingContext2D;
}

function setElementBox(element: HTMLElement, width: number, height: number): void {
  Object.defineProperty(element, "clientWidth", { configurable: true, value: width });
  Object.defineProperty(element, "clientHeight", { configurable: true, value: height });
}

function payloadFixture(): GanttPayload {
  return {
    registries: {
      markers: {
        message_dispatch: { color: "#76FF03", label: "Message Dispatch", symbol: "diamond" },
      },
      spans: {
        llm: { color: "#00E5FF", label: "LLM Call", order: 0 },
        tool: { color: "#FF6D00", label: "Tool Exec", order: 1 },
      },
    },
    traces: [
      {
        id: "trace-1",
        label: "trace-1",
        metadata: {
          elapsed_s: 2,
          instance_id: "demo",
          max_iterations: 2,
          mode: "import",
          model: "test",
          n_actions: 2,
          n_events: 1,
          n_iterations: 1,
          scaffold: "openclaw",
        },
        t0: 1000,
        lanes: [
          {
            agent_id: "agent-1",
            markers: [
              {
                detail: {},
                event: "message_dispatch",
                iteration: 0,
                t: 0.8,
                t_abs: 1000.8,
                t_real: 0.3,
                t_real_abs: 1000.3,
                type: "scheduling",
              },
            ],
            spans: [
              {
                detail: { llm_content: "hello" },
                end: 0.4,
                end_abs: 1000.4,
                end_real: 0.2,
                end_real_abs: 1000.2,
                iteration: 0,
                start: 0,
                start_abs: 1000,
                start_real: 0,
                start_real_abs: 1000,
                type: "llm",
              },
              {
                detail: { tool_name: "bash" },
                end: 1.0,
                end_abs: 1001.0,
                end_real: 0.7,
                end_real_abs: 1000.7,
                iteration: 0,
                start: 0.2,
                start_abs: 1000.2,
                start_real: 0.1,
                start_real_abs: 1000.1,
                type: "tool",
              },
            ],
          },
        ],
      },
    ],
  };
}

describe("CanvasRenderer", () => {
  it("locates and hit-tests a rendered span", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();

    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 10,
      right: 650,
      top: 20,
      width: 640,
      x: 10,
      y: 20,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const renderer = new CanvasRenderer(canvas, wrap);
    const payload = payloadFixture();

    renderer.setPayload(payload);
    (renderer as unknown as { render: () => void }).render();
    const hit = renderer.hitTest(24, 40);
    expect(hit).not.toBeNull();
    if (!hit) {
      throw new Error("expected span hit");
    }
    const card = renderer.locateHit(hit);

    expect(card).not.toBeNull();
    const localHit = renderer.hitTest(card!.x - 10, card!.y - 20);
    expect(localHit?.kind).toBe("span");
    renderer.destroy();
  });

  it("clamps zoom and emits the resulting factor", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();
    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 0,
      right: 640,
      top: 0,
      width: 640,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const renderer = new CanvasRenderer(canvas, wrap);
    const factors: number[] = [];
    renderer.addEventListener("zoom", (event) => {
      factors.push((event as CustomEvent<{ factor: number }>).detail.factor);
    });

    renderer.setZoom(100);

    expect(factors.at(-1)).toBe(32);
    renderer.destroy();
  });

  it("reanchors a span hit after switching to concise layout", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();
    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 0,
      right: 640,
      top: 0,
      width: 640,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const renderer = new CanvasRenderer(canvas, wrap);
    const payload = payloadFixture();

    renderer.setPayload(payload);
    (renderer as unknown as { render: () => void }).render();
    const secondSpan = renderer.hitTest(340, 58);
    expect(secondSpan).not.toBeNull();
    if (!secondSpan) {
      throw new Error("expected second span hit");
    }
    const layered = renderer.locateHit(secondSpan);
    renderer.setViewMode("concise");
    (renderer as unknown as { render: () => void }).render();
    const concise = renderer.locateHit(secondSpan);

    expect(layered).not.toBeNull();
    expect(concise).not.toBeNull();
    expect(concise!.y).toBeLessThan(layered!.y);
    renderer.destroy();
  });

  it("uses compacted coordinates in real mode", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();
    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 0,
      right: 640,
      top: 0,
      width: 640,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const renderer = new CanvasRenderer(canvas, wrap);
    renderer.setPayload(payloadFixture());
    (renderer as unknown as { render: () => void }).render();
    const wallHit = renderer.hitTest(340, 58);
    renderer.setClockMode("real");
    (renderer as unknown as { render: () => void }).render();
    const realHit = renderer.hitTest(260, 58);

    expect(wallHit?.kind).toBe("span");
    expect(realHit?.kind).toBe("span");
    renderer.destroy();
  });

  it("falls back to wall coordinates when real fields are missing", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();
    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 0,
      right: 640,
      top: 0,
      width: 640,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const payload = payloadFixture();
    const span = payload.traces[0].lanes[0].spans[1] as Record<string, unknown>;
    delete span.start_real;
    delete span.end_real;
    delete span.start_real_abs;
    delete span.end_real_abs;

    const renderer = new CanvasRenderer(canvas, wrap);
    renderer.setPayload(payload);
    renderer.setClockMode("real");
    (renderer as unknown as { render: () => void }).render();

    // With stratified layout, tool spans live in the bottom strip. At x=340
    // (time ~0.53 in wall fallback), only the tool span covers that X, which
    // sits at y ≈ 52-62. y=55 lands on the tool row.
    expect(renderer.hitTest(340, 55)?.kind).toBe("span");
    renderer.destroy();
  });

  it("falls back to wall marker coordinates when real marker fields are missing", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();
    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 0,
      right: 640,
      top: 0,
      width: 640,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const payload = payloadFixture();
    const marker = payload.traces[0].lanes[0].markers[0] as Record<string, unknown>;
    delete marker.t_real;
    delete marker.t_real_abs;

    const renderer = new CanvasRenderer(canvas, wrap);
    renderer.setPayload(payload);
    renderer.setClockMode("real");
    (renderer as unknown as { render: () => void }).render();
    const markerBox = (
      renderer as unknown as {
        hitBoxes: Array<{ hit: { kind: string }; x: number; y: number; w: number; h: number }>;
      }
    ).hitBoxes.find((box) => box.hit.kind === "marker");

    expect(markerBox).toBeDefined();
    expect(
      renderer.hitTest(
        markerBox!.x + markerBox!.w / 2,
        markerBox!.y + markerBox!.h / 2,
      )?.kind,
    ).toBe("marker");
    renderer.destroy();
  });

  it("clips resource charts to span bounds with interpolated endpoints", () => {
    const canvas = document.createElement("canvas");
    const wrap = document.createElement("div");
    const context = createMockContext();
    vi.spyOn(canvas, "getContext").mockImplementation(() => context);
    vi.spyOn(canvas, "getBoundingClientRect").mockImplementation(() => ({
      bottom: 360,
      height: 320,
      left: 0,
      right: 640,
      top: 0,
      width: 640,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }));
    setElementBox(canvas, 640, 320);
    setElementBox(wrap, 640, 320);

    const payload = payloadFixture();
    payload.traces[0].lanes[0].markers[0].t = 2;
    payload.traces[0].lanes[0].markers[0].t_abs = 1002;
    payload.traces[0].resource_timeline = [
      { cpu_percent: 10, memory_mb: 100, t: -0.5, t_abs: 999.5 },
      { cpu_percent: 30, memory_mb: 200, t: 0.5, t_abs: 1000.5 },
      { cpu_percent: 50, memory_mb: 300, t: 1.5, t_abs: 1001.5 },
    ];

    const renderer = new CanvasRenderer(canvas, wrap);
    renderer.setPayload(payload);
    (renderer as unknown as { render: () => void }).render();
    const resourceBox = (
      renderer as unknown as {
        hitBoxes: Array<{ hit: { kind: string; timeline?: Array<{ cpu_percent: number; t: number }> } }>;
      }
    ).hitBoxes.find((box) => box.hit.kind === "resource");
    const timeline = resourceBox?.hit.timeline;

    expect(timeline?.map((sample) => sample.t)).toEqual([0, 0.5, 1]);
    expect(timeline?.[0].cpu_percent).toBeCloseTo(20);
    expect(timeline?.[2].cpu_percent).toBeCloseTo(40);
    renderer.destroy();
  });
});
