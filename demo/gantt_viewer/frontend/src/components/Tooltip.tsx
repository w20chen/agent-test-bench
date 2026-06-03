import { Show, createMemo } from "solid-js";

import type { Registries } from "../api/client";
import type { HitCard } from "../canvas/hit";
import { hitAccent, hitRows, hitTitle } from "../canvas/hit";

interface TooltipProps {
  card: HitCard | null;
  onClose: () => void;
  pinned: boolean;
  registries: Registries | null;
}

export default function Tooltip(props: TooltipProps) {
  const position = createMemo(() => {
    if (!props.card || typeof window === "undefined") {
      return {};
    }
    const maxWidth = 540;
    const maxHeight = 420;
    return {
      left: `${Math.min(props.card.x + 16, window.innerWidth - maxWidth - 16)}px`,
      top: `${Math.min(props.card.y + 16, window.innerHeight - maxHeight - 16)}px`,
    };
  });

  return (
    <Show when={props.card}>
      <aside
        class="tooltip-card"
        classList={{ pinned: props.pinned }}
        style={position()}
      >
        <div class="tooltip-head" style={{ color: hitAccent(props.card!.hit, props.registries) }}>
          <strong>{hitTitle(props.card!.hit, props.registries)}</strong>
          <Show when={props.pinned}>
            <button class="tooltip-close" onClick={props.onClose} type="button">
              ×
            </button>
          </Show>
        </div>
        <div class="tooltip-grid">
          {hitRows(props.card!.hit).map(([key, value]) => (
            <>
              <span class="tooltip-key">{key}</span>
              <span class="tooltip-value">{value}</span>
            </>
          ))}
        </div>
        <Show when={props.pinned}>
          <p class="tooltip-hint">Pinned — click canvas or press ESC to clear.</p>
        </Show>
      </aside>
    </Show>
  );
}
