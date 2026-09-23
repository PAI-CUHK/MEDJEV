"""Create the lightweight animated overview embedded in README.md.

This is a presentation asset, not an experimental result. It contains no
patient data, checkpoints, benchmark records, or model outputs.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "jev-runtime-flow.gif"
WIDTH, HEIGHT = 1200, 620


def font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(Path("C:/Windows/Fonts") / filename, size)


def centered(draw, box, text, fnt, fill):
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, align="center", spacing=4)
    x = (left + right - (bounds[2] - bounds[0])) / 2
    y = (top + bottom - (bounds[3] - bounds[1])) / 2 - bounds[1]
    draw.multiline_text((x, y), text, font=fnt, fill=fill, align="center", spacing=4)


def frame(active: int, mode: str):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f7fbfb")
    draw = ImageDraw.Draw(image)
    ink, muted, teal, line = "#103a43", "#668087", "#0b9f95", "#d9e9e9"
    draw.rounded_rectangle((28, 25, WIDTH - 28, HEIGHT - 25), 28, fill="white", outline="#d8e8e8", width=2)
    draw.text((62, 56), "JEV-INSPIRED RUNTIME SEMANTICS", font=font(17, True), fill=teal)
    draw.text((62, 87), "One evidence state. Many questions.", font=font(34, True), fill=ink)
    subtitle = "MEDJEV · clinical text" if mode == "MEDJEV" else "SLEEPJEV · overnight PSG"
    draw.text((WIDTH - 390, 94), subtitle, font=font(18, True), fill=muted)

    labels = [
        ("1", "Shared evidence", "record" if mode == "MEDJEV" else "overnight PSG"),
        ("2", "Runtime question", "query.question" if mode == "MEDJEV" else "SleepQuery"),
        ("3", "Candidate meanings", "supported · contradicted\n· unresolved" if mode == "MEDJEV" else "REM · N2 · event"),
        ("4", "Auditable output", "probabilities + scope"),
    ]
    x0, box_w, gap, top, bottom = 62, 238, 42, 210, 420
    for index, (number, title, detail) in enumerate(labels):
        left = x0 + index * (box_w + gap)
        right = left + box_w
        is_active = index == active
        fill = "#e4f8f4" if is_active else "#fbfdfd"
        outline = teal if is_active else line
        draw.rounded_rectangle((left, top, right, bottom), 18, fill=fill, outline=outline, width=4 if is_active else 2)
        draw.ellipse((left + 18, top + 18, left + 52, top + 52), fill=teal)
        centered(draw, (left + 18, top + 18, left + 52, top + 52), number, font(15, True), "white")
        draw.text((left + 18, top + 76), title, font=font(20, True), fill=ink)
        centered(draw, (left + 15, top + 125, right - 15, bottom - 15), detail, font(16), muted if not is_active else ink)
        if index < len(labels) - 1:
            arrow_x = right + 9
            draw.line((arrow_x, 315, arrow_x + 25, 315), fill=teal if index < active else "#a9bec1", width=4)
            draw.polygon([(arrow_x + 25, 315), (arrow_x + 15, 307), (arrow_x + 15, 323)], fill=teal if index < active else "#a9bec1")

    if active < 3:
        start = x0 + active * (box_w + gap) + box_w + 18
        end = x0 + (active + 1) * (box_w + gap) - 18
        particle_x = start + int((end - start) * (active + 1) / 4)
        draw.ellipse((particle_x - 8, 307, particle_x + 8, 323), fill="#f59e0b")
    else:
        draw.rounded_rectangle((62, 475, WIDTH - 62, 535), 16, fill="#103a43")
        centered(draw, (75, 475, WIDTH - 75, 535), "Meaning follows the runtime candidate identity — not a fixed label position.", font(18, True), "white")
    draw.text((62, 566), "Shared representation  ·  runtime candidate set  ·  inspectable decision", font=font(15), fill=muted)
    return image


def main():
    frames = []
    for mode in ("MEDJEV", "SLEEPJEV"):
        for active in range(4):
            frames.extend([frame(active, mode)] * 5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    print(f"created {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
