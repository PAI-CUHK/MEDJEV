"""Render a JEV-native animated hero: one state, typed questions, decisions."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "jev-runtime-flow.gif"
W, H = 1400, 720


def ft(size, bold=False):
    return ImageFont.truetype(Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf"), size)


def center(draw, box, text, fnt, fill):
    b = draw.multiline_textbbox((0, 0), text, font=fnt, align="center", spacing=4)
    x = (box[0] + box[2] - b[2] + b[0]) / 2
    y = (box[1] + box[3] - b[3] + b[1]) / 2
    draw.multiline_text((x, y), text, font=fnt, fill=fill, align="center", spacing=4)


def dot(base, x, y, color="#f4b94e"):
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rgb = tuple(bytes.fromhex(color[1:]))
    d.ellipse((x - 32, y - 32, x + 32, y + 32), fill=(*rgb, 35))
    d.ellipse((x - 15, y - 15, x + 15, y + 15), fill=(*rgb, 100))
    base.alpha_composite(layer.filter(ImageFilter.GaussianBlur(10)))
    ImageDraw.Draw(base).ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def frame(stage, mode, tick):
    im = Image.new("RGBA", (W, H), "#07171d")
    d = ImageDraw.Draw(im)
    haze = Image.new("RGBA", im.size, (0, 0, 0, 0))
    hd = ImageDraw.Draw(haze)
    hd.ellipse((-190, -180, 510, 500), fill=(11, 159, 149, 70))
    hd.ellipse((920, 290, 1550, 900), fill=(22, 119, 200, 50))
    im.alpha_composite(haze.filter(ImageFilter.GaussianBlur(90)))
    for x in range(25, W, 48): d.line((x, 0, x, H), fill=(255, 255, 255, 8))
    for y in range(25, H, 48): d.line((0, y, W, y), fill=(255, 255, 255, 8))

    white, ink, muted, teal, cyan, line = "#ffffff", "#dff8f5", "#91adb2", "#19c7b4", "#79e9df", "#25464e"
    d.rounded_rectangle((26, 24, W - 26, H - 24), 30, fill=(7, 23, 29, 220), outline=(90, 193, 188, 80), width=2)
    d.text((66, 57), "JEV-INSPIRED TYPED DECISIONS", font=ft(16, True), fill=cyan)
    d.text((66, 91), "One state. Parallel questions. Typed answers.", font=ft(38, True), fill=white)
    d.text((66, 143), "No generated prose  ·  explicit answer space  ·  calibrated probabilities", font=ft(17), fill=muted)
    tag = "MEDJEV / EVIDENCE STATE" if mode == "MEDJEV" else "SLEEPJEV / PSG STATE"
    d.rounded_rectangle((1030, 68, 1328, 112), 22, fill=(25, 199, 180, 35), outline=(25, 199, 180, 140), width=2)
    center(d, (1040, 68, 1318, 112), tag, ft(13, True), cyan)

    cols = [(66, 340), (390, 820), (870, 1334)]
    for left, right in cols:
        d.rounded_rectangle((left, 222, right, 585), 18, fill=(10, 31, 38, 220), outline=line, width=2)
    d.text((92, 250), "STATE", font=ft(14, True), fill=teal)
    d.text((92, 286), "Shared input context", font=ft(23, True), fill=ink)
    state = "Clinical record +\nbiomedical passage" if mode == "MEDJEV" else "Overnight PSG +\ntemporal event index"
    center(d, (90, 340, 316, 438), state, ft(23, True), white)
    d.rounded_rectangle((90, 475, 316, 525), 10, fill=(25, 199, 180, 22), outline=(25, 199, 180, 80), width=1)
    center(d, (100, 475, 306, 525), "encoded once", ft(14, True), cyan)

    d.text((416, 250), "QUESTIONS", font=ft(14, True), fill=teal)
    d.text((416, 286), "Typed questions on the same state", font=ft(23, True), fill=ink)
    questions = [("CHOICE", "Which evidence relation?", "supported  /  contradicted  /  unresolved"), ("SCORE", "How strong is the signal?", "ordered rubric · 1  2  3  4  5"), ("NOUL", "Is the statement supported?", "yes / no · probability of yes")]
    for i, (kind, title, detail) in enumerate(questions):
        y = 337 + i * 70
        active = i == stage % 3
        d.rounded_rectangle((416, y, 794, y + 54), 11, fill=(25, 199, 180, 25 if active else 8), outline=teal if active else line, width=2)
        d.text((432, y + 11), kind, font=ft(11, True), fill=cyan if active else muted)
        d.text((510, y + 9), title, font=ft(14, True), fill=white if active else ink)
        d.text((510, y + 30), detail, font=ft(10), fill=muted)

    d.text((896, 250), "ANSWERS", font=ft(14, True), fill=teal)
    d.text((896, 286), "Decisions software can use", font=ft(23, True), fill=ink)
    outputs = [("choice", "supported", "0.74"), ("score", "3.8 / 5", "0.81"), ("noul", "yes", "0.68")]
    for i, (kind, result, prob) in enumerate(outputs):
        y = 338 + i * 70
        active = i <= stage % 4
        d.text((896, y + 7), kind, font=ft(12, True), fill=cyan if active else muted)
        d.text((974, y + 5), result, font=ft(18, True), fill=white if active else muted)
        d.rounded_rectangle((1092, y + 14, 1296, y + 26), 6, fill=(37, 70, 78, 220))
        if active: d.rounded_rectangle((1092, y + 14, 1092 + int(204 * float(prob)), y + 26), 6, fill=teal)
        d.text((1098, y + 35), f"probability  {prob}" if active else "waiting", font=ft(10), fill=muted)

    if stage < 3:
        x1, x2 = (340, 390) if stage == 0 else ((820, 870) if stage > 0 else (340, 390))
        if stage == 2: x1, x2 = 820, 870
        travel = (tick % 12) / 11
        dot(im, int(x1 + (x2 - x1) * travel), 402)
    else:
        d.rounded_rectangle((66, 618, 1334, 658), 12, fill=(18, 59, 66, 230), outline=(25, 199, 180, 120), width=1)
        center(d, (76, 618, 1324, 658), "JEV pattern: state in  →  named questions in parallel  →  typed probabilities out", ft(16, True), white)
    d.text((66, 681), "MEDJEV / SLEEPJEV", font=ft(12, True), fill=teal)
    d.text((240, 681), "JEV-inspired evidence reasoning for biomedical research", font=ft(12), fill=muted)
    return im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)


def main():
    frames, tick = [], 0
    for mode in ("MEDJEV", "SLEEPJEV"):
        for stage in range(4):
            for _ in range(8):
                frames.append(frame(stage, mode, tick)); tick += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=105, loop=0, optimize=True)
    print(f"created {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__": main()
