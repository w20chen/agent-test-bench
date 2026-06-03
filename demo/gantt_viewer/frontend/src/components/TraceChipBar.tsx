import { For, Show, createMemo } from "solid-js";

import type { TraceDescriptor } from "../api/client";
import { scrollTraceChipGrid } from "./traceChipScroll";

interface TraceChipBarProps {
  descriptors: TraceDescriptor[];
  loadedIds: string[];
  loadingIds: Record<string, boolean>;
  onLoad: (ids: string[]) => Promise<void> | void;
  onUpload: (files: File[]) => Promise<void> | void;
  onRemove: (id: string) => void;
  onToggleVisibility: (id: string) => void;
  snapshotMode: boolean;
  visibility: Record<string, boolean>;
}

export default function TraceChipBar(props: TraceChipBarProps) {
  const loadedSet = createMemo(() => new Set(props.loadedIds));
  const allDescriptorIds = createMemo(() => props.descriptors.map((descriptor) => descriptor.id));
  let fileInputEl!: HTMLInputElement;
  let chipGridEl!: HTMLDivElement;

  function handleFileInputChange(event: Event): void {
    const input = event.currentTarget as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    if (files.length > 0) {
      void props.onUpload(files);
    }
    input.value = "";
  }

  return (
    <section class="trace-strip">
      <span class="trace-strip-header">TRACES</span>
      <div
        class="trace-chip-grid"
        onWheel={(event) => {
          if (
            scrollTraceChipGrid(
              chipGridEl,
              event.deltaX,
              event.deltaY,
            )
          ) {
            event.preventDefault();
          }
        }}
        ref={chipGridEl}
      >
        <For each={props.descriptors}>
          {(descriptor) => {
            const loaded = () => loadedSet().has(descriptor.id);
            const visible = () => props.visibility[descriptor.id] !== false;
            return (
              <div
                class="trace-chip"
                classList={{
                  hidden: loaded() && !visible(),
                  loading: props.loadingIds[descriptor.id] === true,
                  loaded: loaded(),
                }}
              >
                <button
                  class="trace-chip-main"
                  title={`${descriptor.label} · ${descriptor.source_format}`}
                  onClick={() =>
                    loaded()
                      ? props.onToggleVisibility(descriptor.id)
                      : props.onLoad([descriptor.id])
                  }
                  type="button"
                >
                  {descriptor.label}
                </button>
                <Show when={loaded()}>
                  <button
                    class="trace-remove"
                    onClick={() => props.onRemove(descriptor.id)}
                    type="button"
                  >
                    ×
                  </button>
                </Show>
              </div>
            );
          }}
        </For>
      </div>
      <Show when={!props.snapshotMode}>
        <div class="trace-actions">
          <button
            class="secondary-btn"
            onClick={() => fileInputEl.click()}
            type="button"
          >
            + JSONL
          </button>
          <button
            class="primary-btn"
            onClick={() => props.onLoad(allDescriptorIds())}
            type="button"
          >
            Load all
          </button>
        </div>
        <input
          accept=".jsonl"
          class="visually-hidden"
          multiple
          onChange={handleFileInputChange}
          ref={fileInputEl}
          type="file"
        />
      </Show>
    </section>
  );
}
