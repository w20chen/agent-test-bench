import { Show, createSignal, onCleanup, onMount } from "solid-js";

interface DropZoneProps {
  enabled: boolean;
  onUpload: (files: File[]) => Promise<void>;
}

export default function DropZone(props: DropZoneProps) {
  const [active, setActive] = createSignal(false);
  let dragDepth = 0;

  function resetDragState(): void {
    dragDepth = 0;
    setActive(false);
  }

  function jsonlFilesFromDrop(event: DragEvent): File[] {
    return Array.from(event.dataTransfer?.files ?? []).filter((file) =>
      file.name.toLowerCase().endsWith(".jsonl"),
    );
  }

  async function onDrop(event: DragEvent) {
    event.preventDefault();
    resetDragState();
    const files = jsonlFilesFromDrop(event);
    if (files.length > 0) {
      await props.onUpload(files);
    }
  }

  function onDragEnter(event: DragEvent) {
    event.preventDefault();
    dragDepth += 1;
    setActive(true);
  }

  function onDragLeave(event: DragEvent) {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) {
      setActive(false);
    }
  }

  function onDragOver(event: DragEvent) {
    event.preventDefault();
    setActive(true);
  }

  onMount(() => {
    if (!props.enabled) {
      return;
    }
    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
  });

  onCleanup(() => {
    if (!props.enabled) {
      return;
    }
    window.removeEventListener("dragenter", onDragEnter);
    window.removeEventListener("dragleave", onDragLeave);
    window.removeEventListener("dragover", onDragOver);
    window.removeEventListener("drop", onDrop);
  });

  return (
    <Show when={props.enabled && active()}>
      <div class="dropzone-overlay">
        <div class="dropzone-card">
          <p class="eyebrow">Drop JSONL</p>
          <strong>Release to upload and render the trace immediately.</strong>
        </div>
      </div>
    </Show>
  );
}
