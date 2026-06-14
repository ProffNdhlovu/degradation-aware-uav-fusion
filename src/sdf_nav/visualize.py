from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np

from sdf_nav.evaluate import evaluate_run, run_fusion


COLORS = {
    "truth": "#111827",
    "fused": "#2563eb",
    "gnss": "#f97316",
    "vio": "#16a34a",
    "grid": "#e5e7eb",
    "text": "#111827",
}


def finite_points(points: np.ndarray) -> np.ndarray:
    return points[np.isfinite(points).all(axis=1)]


def polyline(points: np.ndarray, project) -> str:
    finite = finite_points(points)
    if len(finite) == 0:
        return ""
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in (project(point) for point in finite))


def make_projector(series: list[np.ndarray], width: int, height: int, pad: int = 56):
    valid = [finite_points(points) for points in series if len(finite_points(points))]
    all_points = np.vstack(valid)
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])

    def project(point: np.ndarray) -> tuple[float, float]:
        x = pad + (point[0] - min_xy[0]) * scale
        y = height - pad - (point[1] - min_xy[1]) * scale
        return float(x), float(y)

    return project, min_xy, max_xy, scale


def draw_route_svg(data: dict[str, np.ndarray], fused: np.ndarray, metrics: dict[str, object]) -> str:
    width = 1000
    height = 760
    project, min_xy, max_xy, _ = make_projector(
        [data["truth"], data["gnss"], data["vio"], fused],
        width,
        height,
    )
    truth_line = polyline(data["truth"], project)
    fused_line = polyline(fused, project)
    gnss_line = polyline(data["gnss"], project)
    vio_line = polyline(data["vio"], project)
    start_x, start_y = project(data["truth"][0])
    end_x, end_y = project(data["truth"][-1])
    subtitle = (
        f"Fused RMSE {metrics['rmse_fused_m']:.2f} m | "
        f"GNSS {metrics['rmse_gnss_m']:.2f} m | "
        f"VIO {metrics['rmse_vio_m']:.2f} m"
        if np.isfinite(metrics["rmse_vio_m"])
        else f"Fused RMSE {metrics['rmse_fused_m']:.2f} m | GNSS {metrics['rmse_gnss_m']:.2f} m | VIO missing"
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="56" y="42" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="{COLORS['text']}">INSANE Navigation Fusion Replay</text>
  <text x="56" y="72" font-family="Arial, sans-serif" font-size="15" fill="#4b5563">{html.escape(subtitle)}</text>
  <text x="56" y="724" font-family="Arial, sans-serif" font-size="13" fill="#6b7280">Bounds: x {min_xy[0]:.1f}..{max_xy[0]:.1f} m, y {min_xy[1]:.1f}..{max_xy[1]:.1f} m</text>
  <g stroke="{COLORS['grid']}" stroke-width="1">
    {"".join(f'<line x1="{x}" y1="100" x2="{x}" y2="690"/>' for x in range(100, 951, 100))}
    {"".join(f'<line x1="56" y1="{y}" x2="944" y2="{y}"/>' for y in range(120, 681, 80))}
  </g>
  <polyline points="{gnss_line}" fill="none" stroke="{COLORS['gnss']}" stroke-width="2" opacity="0.65"/>
  <polyline points="{vio_line}" fill="none" stroke="{COLORS['vio']}" stroke-width="2" opacity="0.65"/>
  <polyline points="{truth_line}" fill="none" stroke="{COLORS['truth']}" stroke-width="4" opacity="0.9"/>
  <polyline points="{fused_line}" fill="none" stroke="{COLORS['fused']}" stroke-width="4" opacity="0.9"/>
  <circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="7" fill="#10b981"/>
  <circle cx="{end_x:.2f}" cy="{end_y:.2f}" r="7" fill="#ef4444"/>
  <g font-family="Arial, sans-serif" font-size="14">
    <rect x="724" y="32" width="220" height="116" fill="#ffffff" stroke="#d1d5db"/>
    <line x1="744" y1="58" x2="794" y2="58" stroke="{COLORS['truth']}" stroke-width="4"/><text x="808" y="63" fill="{COLORS['text']}">Ground truth</text>
    <line x1="744" y1="84" x2="794" y2="84" stroke="{COLORS['fused']}" stroke-width="4"/><text x="808" y="89" fill="{COLORS['text']}">Fused estimate</text>
    <line x1="744" y1="110" x2="794" y2="110" stroke="{COLORS['gnss']}" stroke-width="3"/><text x="808" y="115" fill="{COLORS['text']}">GNSS</text>
    <line x1="744" y1="136" x2="794" y2="136" stroke="{COLORS['vio']}" stroke-width="3"/><text x="808" y="141" fill="{COLORS['text']}">VIO / odom</text>
  </g>
</svg>
"""


def draw_animation_html(data: dict[str, np.ndarray], fused: np.ndarray, metrics: dict[str, object]) -> str:
    width = 1000
    height = 760
    project, _, _, _ = make_projector([data["truth"], data["gnss"], data["vio"], fused], width, height)
    frame_count = min(220, len(data["t"]))
    idx = np.linspace(0, len(data["t"]) - 1, frame_count).astype(int)

    def frames(points: np.ndarray) -> str:
        coords = []
        for i in idx:
            point = points[i]
            if np.isfinite(point).all():
                x, y = project(point)
                coords.append([round(x, 2), round(y, 2)])
            else:
                coords.append(None)
        return str(coords).replace("None", "null")

    svg = draw_route_svg(data, fused, metrics)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>INSANE Navigation Simulation</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f8fafc; color: #111827; }}
    main {{ max-width: 1100px; margin: 24px auto; padding: 0 20px; }}
    .panel {{ background: white; border: 1px solid #d1d5db; padding: 16px; }}
    svg {{ width: 100%; height: auto; display: block; }}
    button {{ padding: 8px 12px; border: 1px solid #9ca3af; background: white; cursor: pointer; }}
  </style>
</head>
<body>
<main>
  <div class="panel">
    {svg}
    <p><button id="toggle">Pause</button> <span id="time"></span></p>
  </div>
</main>
<script>
const truth = {frames(data["truth"])};
const fused = {frames(fused)};
const gnss = {frames(data["gnss"])};
const vio = {frames(data["vio"])};
const times = {[round(float(data["t"][i]), 2) for i in idx]};
const svg = document.querySelector("svg");
function marker(id, color, r) {{
  const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  c.setAttribute("id", id);
  c.setAttribute("r", r);
  c.setAttribute("fill", color);
  c.setAttribute("stroke", "white");
  c.setAttribute("stroke-width", "2");
  svg.appendChild(c);
  return c;
}}
const markers = [
  [marker("truthDot", "{COLORS['truth']}", 7), truth],
  [marker("fusedDot", "{COLORS['fused']}", 7), fused],
  [marker("gnssDot", "{COLORS['gnss']}", 5), gnss],
  [marker("vioDot", "{COLORS['vio']}", 5), vio],
];
let i = 0;
let running = true;
function tick() {{
  for (const [dot, series] of markers) {{
    const p = series[i];
    dot.style.display = p ? "block" : "none";
    if (p) {{
      dot.setAttribute("cx", p[0]);
      dot.setAttribute("cy", p[1]);
    }}
  }}
  document.getElementById("time").textContent = `t = ${{times[i]}} s`;
  if (running) i = (i + 1) % truth.length;
}}
document.getElementById("toggle").onclick = () => {{
  running = !running;
  document.getElementById("toggle").textContent = running ? "Pause" : "Play";
}};
setInterval(tick, 60);
tick();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()

    loaded = np.load(args.input, allow_pickle=False)
    data = {key: loaded[key] for key in loaded.files}
    fused, _ = run_fusion(data)
    metrics = evaluate_run(data)

    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(draw_route_svg(data, fused, metrics), encoding="utf-8")
    args.html.write_text(draw_animation_html(data, fused, metrics), encoding="utf-8")
    print(f"wrote {args.svg}")
    print(f"wrote {args.html}")


if __name__ == "__main__":
    main()

