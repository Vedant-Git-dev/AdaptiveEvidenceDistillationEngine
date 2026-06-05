type Props = {
  values: number[];
  width?: number;
  height?: number;
  className?: string;
};

/** Tiny inline-SVG sparkline. No axes, no labels — just a shape. */
export function Sparkline({ values, width = 120, height = 32, className }: Props) {
  if (!values.length) {
    return (
      <svg
        width={width}
        height={height}
        className={className}
        viewBox={`0 0 ${width} ${height}`}
      />
    );
  }
  const pad = 2;
  const w = width - pad * 2;
  const h = height - pad * 2;
  const min = 0;
  const max = 1; // coverage is always 0..1
  const xStep = values.length === 1 ? 0 : w / (values.length - 1);
  const pts = values
    .map((v, i) => {
      const x = pad + i * xStep;
      const y = pad + h - ((v - min) / (max - min)) * h;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg
      width={width}
      height={height}
      className={className}
      viewBox={`0 0 ${width} ${height}`}
    >
      <polyline
        points={pts}
        fill="none"
        stroke="currentColor"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
