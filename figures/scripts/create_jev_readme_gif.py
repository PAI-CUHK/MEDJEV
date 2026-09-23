"""Render the dark, animated architecture hero embedded in README.md."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "jev-runtime-flow.gif"
WIDTH, HEIGHT = 1400, 720


def font(size: int, bold: bool = False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(Path("C:/Windows/Fonts") / name, size)


def centered(draw, box, text, fnt, fill):
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=fnt, align="center", spacing=5)
    x = (left + right - bounds[2] + bounds[0]) / 2
    y = (top + bottom - bounds[3] + bounds[1]) / 2
    draw.multiline_text((x, y), text, font=fnt, fill=fill, align="center", spacing=5)


def glow_dot(base, x, y, color="#2de0cb"):
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    rgb = tuple(bytes.fromhex(color[1:]))
    for radius, alpha in ((30, 20), (18, 45), (10, 100)):
        glow_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*rgb, alpha))
    base.alpha_composite(glow.filter(ImageFilter.GaussianBlur(8)))
    ImageDraw.Draw(base).ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def frame(active: int, mode: str, tick: int):
    image = Image.new("RGBA", (WIDTH, HEIGHT), "#07171d")
    draw = ImageDraw.Draw(image)
    atmosphere = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ad = ImageDraw.Draw(atmosphere)
    ad.ellipse((-170, -170, 560, 510), fill=(11, 159, 149, 70))
    ad.ellipse((850, 300, 1570, 930), fill=(22, 119, 200, 46))
    image.alpha_composite(atmosphere.filter(ImageFilter.GaussianBlur(85)))
    for x in range(24, WIDTH, 48):
        draw.line((x, 0, x, HEIGHT), fill=(255, 255, 255, 8), width=1)
    for y in range(24, HEIGHT, 48):
        draw.line((0, y, WIDTH, y), fill=(255, 255, 255, 8), width=1)

    ink, white, muted, teal, cyan, line = "#dff8f5", "#ffffff", "#91adb2", "#19c7b4", "#79e9df", "#25464e"
    draw.rounded_rectangle((26, 24, WIDTH - 26, HEIGHT - 24), 30, fill=(7, 23, 29, 212), outline=(90, 193, 188, 80), width=2)
    draw.text((66, 58), "JEV-INSPIRED RUNTIME SEMANTICS", font=font(16, True), fill=cyan)
    draw.text((66, 91), "Runtime semantics, visualized.", font=font(42, True), fill=white)
    draw.text((66, 145), "Shared evidence representation  ·  explicit candidate meaning  ·  inspectable decision", font=font(17), fill=muted)
    pill = "MEDJEV  /  CLINICAL TEXT" if mode == "MEDJEV" else "SLEEPJEV  /  OVERNIGHT PSG"
    draw.rounded_rectangle((1030, 72, 1328, 116), 22, fill=(11, 159, 149, 38), outline=(25, 199, 180, 130), width=2)
    centered(draw, (1040, 72, 1318, 116), pill, font(13, True), cyan)

    labels = [
        ("01", "SHARED EVIDENCE", "record" if mode == "MEDJEV" else "overnight PSG", "encode once"),
        ("02", "RUNTIME QUESTION", "query.question" if mode == "MEDJEV" else "SleepQuery", "ask at runtime"),
        ("03", "CANDIDATE MEANINGS", "supported · contradicted\n· unresolved" if mode == "MEDJEV" else "REM · N2 · event", "meaning is explicit"),
        ("04", "AUDITABLE OUTPUT", "probabilities + scope", "decision is inspectable"),
    ]
    x0, box_w, gap, top, bottom = 66, 274, 34, 250, 492
    for index, (number, title, detail, caption) in enumerate(labels):
        left = x0 + index * (box_w + gap)
        right = left + box_w
        is_active = index == active
        fill = (14, 45, 52, 240) if is_active else (10, 31, 38, 220)
        outline = (25, 199, 180, 240) if is_active else (37, 70, 78, 220)
        if is_active:
            glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow)
            gd.rounded_rectangle((left - 7, top - 7, right + 7, bottom + 7), 23, outline=(25, 199, 180, 95), width=8)
            image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(13)))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((left, top, right, bottom), 18, fill=fill, outline=outline, width=3 if is_active else 2)
        draw.line((left + 1, top + 3, right - 1, top + 3), fill=teal if is_active else line, width=4)
        draw.text((left + 22, top + 24), number, font=font(14, True), fill=teal if is_active else muted)
        draw.text((left + 22, top + 66), title, font=font(17, True), fill=ink)
        centered(draw, (left + 17, top + 112, right - 17, bottom - 53), detail, font(19, is_active), white if is_active else muted)
        draw.text((left + 22, bottom - 32), caption, font=font(12), fill=teal if is_active else muted)
        if index < len(labels) - 1:
            ax = right + 10
            color = teal if index < active else line
            draw.line((ax, 370, ax + 20, 370), fill=color, width=3)
            draw.polygon([(ax + 30, 370), (ax + 19, 363), (ax + 19, 377)], fill=color)

    if active < 3:
        start = x0 + active * (box_w + gap) + box_w + 15
        end = x0 + (active + 1) * (box_w + gap) - 15
        travel = (tick % 12) / 11
        glow_dot(image, int(start + (end - start) * travel), 370, "#f4b94e")
    else:
        draw.rounded_rectangle((66, 557, WIDTH - 66, 614), 16, fill=(18, 59, 66, 225), outline=(25, 199, 180, 130), width=1)
        centered(draw, (82, 557, WIDTH - 82, 614), "Meaning follows runtime candidate identity — not a fixed label position.", font(18, True), white)
    draw.text((66, 656), "MEDJEV / SLEEPJEV", font=font(13, True), fill=teal)
    draw.text((240, 656), "JEV-inspired evidence formation → semantic decision separation", font=font(13), fill=muted)
    return image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)


def main():
    frames = []
    tick = 0
    for mode in ("MEDJEV", "SLEEPJEV"):
        for active in range(4):
            for _ in range(8):
                frames.append(frame(active, mode, tick))
                tick += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=105, loop=0, optimize=True)
    print(f"created {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
