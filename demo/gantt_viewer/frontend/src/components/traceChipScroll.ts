export function scrollTraceChipGrid(
  element: Pick<HTMLElement, "scrollWidth" | "clientWidth" | "scrollLeft">,
  deltaX: number,
  deltaY: number,
): boolean {
  if (element.scrollWidth <= element.clientWidth) {
    return false;
  }

  const dominantDelta = Math.abs(deltaX) > Math.abs(deltaY) ? deltaX : deltaY;
  if (dominantDelta === 0) {
    return false;
  }

  element.scrollLeft += dominantDelta;
  return true;
}
