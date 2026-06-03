import { For, Show } from "solid-js";

import type { Registries } from "../api/client";
import { displayColor } from "../theme/displayColor";

interface LegendProps {
  registries: Registries | null;
}

export default function Legend(props: LegendProps) {
  return (
    <section class="legend-card">
      <span class="trace-strip-header">LEGEND</span>
      <Show when={props.registries}>
        <For each={Object.entries(props.registries?.spans ?? {})}>
          {([key, value]) => (
            <div class="legend-item" title={key}>
              <span class="legend-swatch" style={{ background: displayColor(value.color) }} />
              <span>{value.label}</span>
            </div>
          )}
        </For>
      </Show>
    </section>
  );
}
