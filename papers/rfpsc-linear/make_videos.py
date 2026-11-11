"""Render two captioned, silent explainers from the checked-in evidence."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from imageio_ffmpeg import write_frames

HERE = Path(__file__).resolve().parent
SIZE = (960, 540)
FPS = 12
SCENE_SECONDS = 8
BG = "#142e25"
PANEL = "#213e33"
FG = "#eff7eb"
MUTED = "#bacfc1"
LIME = "#d8ee95"
GRID = "#466457"

ALGORITHM_CAPTIONS = [
    "常规 PSC 将有功功率误差乘以同步增益，积分后更新功角。",
    "RFPSC 在同一反馈环之外，将功率参考直接前馈到有源电阻支路。",
    "空载、小信号且电网纯感性时，复极点对被相消。功率响应为一阶。",
]


def write_video(kind, english, title, captions, out, font_path, cases, summary):
    filename = kind + ("-en" if english else "")
    title_font = ImageFont.truetype(font_path, 36)
    body_font = ImageFont.truetype(font_path, 25)
    small_font = ImageFont.truetype(font_path, 18)
    writer = write_frames(str(out / (filename + ".mp4")), SIZE, fps=FPS, codec="libx264", macro_block_size=2, output_params=["-movflags", "+faststart"])
    writer.send(None)
    try:
        for scene, caption in enumerate(captions):
            for frame in range(SCENE_SECONDS * FPS):
                image = Image.new("RGB", SIZE, BG)
                draw = ImageDraw.Draw(image)
                draw.text((52, 26), "ReproGrid  /  RFPSC", font=small_font, fill=MUTED)
                draw.text((52, 65), title, font=title_font, fill=FG)
                draw.rounded_rectangle((52, 125, 908, 435), radius=16, fill=PANEL)
                if kind == "algorithm":
                    algorithm_scene(draw, scene, english, body_font, small_font)
                else:
                    record = summary[scene]
                    result_scene(draw, cases[record["id"]], record, english, body_font, small_font)
                draw.text((52, 458), caption, font=small_font, fill=FG)
                draw.line((52, 508, 908, 508), fill=GRID, width=2)
                elapsed = scene * SCENE_SECONDS * FPS + frame
                draw.line((52, 508, 52 + 856 * (elapsed + 1) / (len(captions) * SCENE_SECONDS * FPS), 508), fill=LIME, width=3)
                if scene == 0 and frame == 0:
                    image.save(out / (filename + ".png"), optimize=True)
                writer.send(image.tobytes())
    finally:
        writer.close()

    def stamp(index):
        return f"00:00:{index * SCENE_SECONDS:02d}.000"

    cues = [f"{stamp(index)} --> {stamp(index + 1)}\n{caption}" for index, caption in enumerate(captions)]
    (out / (filename + ".vtt")).write_text("WEBVTT\n\n" + "\n\n".join(cues) + "\n", encoding="utf-8")


def algorithm_scene(draw, scene, english, body, small):
    nodes = [(95, "P_ref − P"), (361, "K_p / s"), (627, "Power angle θ" if english else "功角 θ")]
    for left, name in nodes:
        draw.rounded_rectangle((left, 173, left + 217, 236), radius=10, outline=LIME, width=2)
        draw.text((left + 108, 204), name, font=body, fill=FG, anchor="mm")
    for left in (321, 587):
        draw.line((left, 204, left + 32, 204), fill=LIME, width=3)
        draw.polygon([(left + 32, 204), (left + 22, 198), (left + 22, 210)], fill=LIME)

    if scene >= 1:
        draw.rounded_rectangle((95, 268, 844, 325), radius=10, fill="#345445")
        draw.text((470, 296), "P_ref / V   →   d-axis current reference   →   active resistance" if english else "P_ref / V   →   d 轴电流参考   →   有源电阻支路", font=body, fill=LIME, anchor="mm")
    if scene == 2:
        draw.text((95, 360), "No-load normalized response   1 − exp(−ατ)      α = Rₐ / L" if english else "空载归一化响应   1 − exp(−ατ)      α = Rₐ / L", font=body, fill=FG)
    elif scene == 0:
        draw.text((95, 285), "Feedback tracks the active-power reference" if english else "反馈环跟踪有功功率参考", font=body, fill=MUTED)
    else:
        draw.text((95, 360), "The feedback gain is the same with feedforward enabled." if english else "加入参考前馈后，反馈增益保持相同。", font=small, fill=MUTED)


def result_scene(draw, points, record, english, body, small):
    draw.text((83, 143), f"L = {record['inductance_pu']:.1f} p.u.", font=body, fill=FG)
    draw.text((467, 148), "RFPSC", font=small, fill=LIME)
    draw.text((564, 148), "Conventional PSC" if english else "常规 PSC", font=small, fill=MUTED)
    x = lambda value: 102 + value * 9.4
    y = lambda value: 393 - value * 172
    for value in (0, .25, .5, .75, 1):
        draw.line((102, y(value), 854, y(value)), fill=GRID)
        draw.text((73, y(value) - 8), f"{value:g}", font=small, fill=MUTED)
    for time in (0, 20, 40, 60, 80):
        draw.line((x(time), 194, x(time), 393), fill=GRID)
        draw.text((x(time) - 8, 398), str(time), font=small, fill=MUTED)
    for column, color in ((1, MUTED), (2, LIME)):
        draw.line([(x(point[0]), y(point[column])) for point in points if point[0] <= 80], fill=color, width=4)
    draw.text((870, 397), "ms", font=small, fill=MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", required=True)
    args = parser.parse_args()
    results = HERE / "results"
    cases = json.loads((results / "chart.json").read_text())["cases"]
    summary = json.loads((results / "summary.json").read_text())["cases"]
    output = HERE / "media"
    output.mkdir(exist_ok=True)
    strong, intermediate, weak = summary
    results_captions = [
        f"强网 L = {strong['inductance_pu']:.1f} p.u.：RFPSC 用时 {strong['rfpsc']['rise_time_ms']:.2f} ms，PSC 为 {strong['psc']['rise_time_ms']:.2f} ms。",
        f"L = {intermediate['inductance_pu']:.1f} p.u.：RFPSC 消除了 {intermediate['psc']['overshoot_percent']:.2f}% 的超调，但上升时间较长。",
        f"L = {weak['inductance_pu']:.1f} p.u.：本次是空载小信号计算。论文的硬件波形仍待复核。",
    ]
    algorithm_en = [
        "Conventional PSC integrates the power error times the synchronizing gain.",
        "RFPSC adds a direct path from the power reference to the active-resistance branch.",
        "At no load with a purely inductive grid, the complex poles cancel in the small-signal response.",
    ]
    results_en = [
        f"Strong grid: RFPSC rises in {strong['rfpsc']['rise_time_ms']:.2f} ms. Conventional PSC takes {strong['psc']['rise_time_ms']:.2f} ms.",
        f"At {intermediate['inductance_pu']:.1f} p.u., RFPSC removes {intermediate['psc']['overshoot_percent']:.2f}% overshoot, with a longer rise time.",
        f"At {weak['inductance_pu']:.1f} p.u., this is a no-load small-signal calculation. Hardware traces await verification.",
    ]
    for english, report_file, algorithm, results_text in (
        (False, "report.json", ALGORITHM_CAPTIONS, results_captions),
        (True, "report.en.json", algorithm_en, results_en),
    ):
        videos = json.loads((HERE / report_file).read_text(encoding="utf-8"))["videos"]
        for video, captions in zip(videos, (algorithm, results_text)):
            write_video(video["kind"], english, video["title"], captions, output, args.font, cases, summary)
    icon = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)
    draw.rounded_rectangle((0, 0, 63, 63), radius=16, fill="#184134")
    for left, top, color in ((17, 17, LIME), (35, 17, "#829e62"), (17, 35, "#547253"), (35, 35, LIME)):
        draw.rounded_rectangle((left, top, left + 12, top + 12), radius=2, fill=color)
    icon.save(HERE.parent.parent / "site" / "icon.png")


if __name__ == "__main__":
    main()
