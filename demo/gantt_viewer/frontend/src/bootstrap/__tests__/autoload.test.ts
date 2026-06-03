import type { TraceDescriptor } from "../../api/client";

import { selectInitialTraceIds, shouldAutoloadAll } from "../autoload";

function makeDescriptor(id: string): TraceDescriptor {
  return {
    id,
    label: id,
    mtime: 0,
    path: `/tmp/${id}.jsonl`,
    size_bytes: 1,
    source_format: "trace",
  };
}

describe("autoload bootstrap", () => {
  it("keeps the default single-trace startup when the query param is absent", () => {
    expect(selectInitialTraceIds([makeDescriptor("a"), makeDescriptor("b")], "")).toEqual(["a"]);
  });

  it("loads all traces when autoload=all is present", () => {
    expect(selectInitialTraceIds([makeDescriptor("a"), makeDescriptor("b")], "?autoload=all")).toEqual([
      "a",
      "b",
    ]);
  });

  it("ignores other autoload values", () => {
    expect(shouldAutoloadAll("?autoload=first")).toBe(false);
  });
});
