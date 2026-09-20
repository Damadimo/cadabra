"""Drive the demo in a real browser: click everything, and check the numbers on screen against the API.

  uv run --with playwright python scripts/ui_check.py [--url http://127.0.0.1:8000] [--race]

Fails loudly on a console error, a failed request, an empty 3D canvas, or any figure that does not match what
/api/compare, /api/parts and /api/scoreboard return. --race also runs one live race (costs a few cents of API credit).
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from playwright.sync_api import sync_playwright

PROBLEMS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{'' if ok else ': ' + detail}")
    if not ok:
        PROBLEMS.append(f"{label}: {detail}")


def num(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(m.group()) if m else None


def canvas_has_content(page, selector: str) -> bool:
    """True if the element's pixels are not one flat colour (three.js does not preserve the drawing buffer, so
    reading the WebGL buffer afterwards returns nothing: screenshot the element instead)."""
    import io

    from PIL import Image

    el = page.query_selector(selector)
    if el is None:
        return False
    img = Image.open(io.BytesIO(el.screenshot())).convert("RGB").resize((120, 90))
    colours = {img.getpixel((x, y)) for x in range(0, 120, 3) for y in range(0, 90, 3)}
    return len(colours) > 6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--race", action="store_true", help="also run one live race (spends API credit)")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=not args.headed)
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        errors: list[str] = []
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") and "favicon" not in m.text else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: errors.append(f"request failed: {r.url}"))
        page.on("response", lambda r: errors.append(f"HTTP {r.status}: {r.url}") if r.status >= 400 else None)
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_timeout(1500)

        print("\ncompare tab")
        part_id = page.eval_on_selector("#parts", "e => e.value")
        data = page.evaluate("id => fetch('/api/compare/' + encodeURIComponent(id)).then(r => r.json())", part_id)
        lanes = data["lanes"]
        check(bool(part_id), "a part is selected", part_id)
        check(page.eval_on_selector("#sheet", "i => i.naturalWidth > 0"), "drawing sheet image loads")
        bbox_text = page.inner_text("#bboxline")
        for v in data["bbox"]:
            check(f"{v:.4f}" in bbox_text, f"bounding box {v:.4f} shown", bbox_text)
        check(str(data["tier_faces"]) in bbox_text, "face count shown", bbox_text)

        page.click("#modelBtn")
        page.wait_for_timeout(300)
        items = page.query_selector_all("#modelMenu button")
        check(len(items) == len(lanes) + 1, "the model menu lists every model plus the reference", f"{len(items)} items, {len(lanes)} lanes")
        for i, lane in enumerate(lanes):
            text = items[i + 1].inner_text()
            if lane.get("iou_aligned") is not None:
                check(f"{lane['iou_aligned']:.3f}" in text, f"menu IoU matches the API for {lane['label']}", text)
            check(("✓" in text) == bool(lane["success"]), f"menu verdict matches for {lane['label']}", text)
        page.keyboard.press("Escape")

        # every model: pick it from the menu, confirm the stage, stats and program follow
        for i in range(len(lanes) + 1):
            page.click("#modelBtn")
            page.wait_for_timeout(250)
            page.query_selector_all("#modelMenu button")[i].click()
            page.wait_for_timeout(900)
            stats = page.inner_text("#stagestats")
            code = page.locator("#stagecode").text_content() or ""      # the program panel is a slide-over
            owner = page.locator("#codeowner").text_content() or ""
            if i == 0:
                check(canvas_has_content(page, "#stage"), "reference renders in the stage")
                check(code.strip().startswith("import cadquery"), "reference program shown", code[:40])
                check("reference" in owner, "program panel names the reference", owner)
                continue
            lane = lanes[i - 1]
            check(lane["label"] in page.inner_text("#modelBtnText"), f"dropdown shows {lane['label']}", page.inner_text("#modelBtnText"))
            if lane.get("iou_aligned") is not None:
                check(f"{lane['iou_aligned']:.3f}" in stats, f"stage IoU matches for {lane['label']}", stats)
            check(("correct" in stats) == bool(lane["success"]), f"stage verdict matches for {lane['label']}", stats)
            if lane.get("output_tokens") is not None:
                check(str(lane["output_tokens"]) in stats, f"token count matches for {lane['label']}", stats)
            if lane.get("code"):
                check(code.strip()[:60] == lane["code"].strip()[:60], f"program matches for {lane['label']}", code[:60])
                check(lane["label"] in owner, f"program panel names {lane['label']}", owner)
            if lane.get("mesh"):
                check(canvas_has_content(page, "#stage"), f"answer renders for {lane['label']}")

        print("\nstage contents")
        shown = page.evaluate("window.__stage()")
        check(len(shown) == 2, "overlay draws the prediction and the reference, and nothing else", str(len(shown)))
        if len(shown) == 2:
            pred, ref = shown
            check(ref["opacity"] < 1 and pred["opacity"] == 1, "the reference is the translucent one", str([pred["opacity"], ref["opacity"]]))
            lane = next((x for x in lanes if x.get("iou_aligned") is not None), None)
            if lane and lane["iou_aligned"] > 0.95:  # the metric aligns the prediction, so the two should sit on top of each other
                gap = max(abs(a - b) for a, b in zip(pred["lo"] + pred["hi"], ref["lo"] + ref["hi"]))
                size = max(h - l for h, l in zip(ref["hi"], ref["lo"]))
                check(gap < 0.08 * size, f"a {lane['iou_aligned']:.3f} IoU prediction lands on the reference", f"corners differ by {gap:.4f} of {size:.4f}")
        # the stage rotates, so the framing has to hold at every angle, not just the one it starts at
        worst = 0.0
        for _ in range(10):
            page.wait_for_timeout(1300)
            ndc = page.evaluate("window.__fit()")["ndc"]
            worst = max(worst, max(abs(v) for pair in ndc for v in pair))
        check(worst < 1.0, "the part stays inside the frame through a full rotation", f"reaches {worst:.3f} of the frustum")

        for _ in range(8):  # fast part changes: a mesh from a cancelled load must not be left in the scene
            page.click("#next")
            page.wait_for_timeout(110)
        page.wait_for_timeout(2500)
        check(len(page.evaluate("window.__stage()")) == 2, "no mesh left behind after fast part changes", str(page.evaluate("window.__stage()")))
        for _ in range(8):
            page.click("#prev")
            page.wait_for_timeout(110)
        page.wait_for_timeout(2500)

        print("\ncontrols")
        page.uncheck("#overlay")
        page.wait_for_timeout(700)
        check("prediction only" in page.inner_text("#stagehint"), "overlay off changes the hint", page.inner_text("#stagehint"))
        check(len(page.evaluate("window.__stage()")) == 1, "overlay off leaves only the prediction")
        page.check("#overlay")
        page.wait_for_timeout(700)
        check("reference" in page.inner_text("#stagehint"), "overlay on changes the hint", page.inner_text("#stagehint"))
        page.uncheck("#spin")
        check(not page.evaluate("document.querySelector('#spin').checked"), "rotate toggles")
        page.check("#spin")

        before = page.eval_on_selector("#parts", "e => e.value")
        page.click("#next")
        page.wait_for_timeout(1400)
        after = page.eval_on_selector("#parts", "e => e.value")
        check(after != before, "next moves to another part", f"{before} -> {after}")
        page.click("#prev")
        page.wait_for_timeout(1400)
        check(page.eval_on_selector("#parts", "e => e.value") == before, "previous comes back")
        check(page.inner_text("#stagestats").strip() != "", "stats redraw after moving")

        print("\nprogram panel")
        page.click("#progBtn")
        page.wait_for_timeout(500)
        check(page.is_visible("#progdrawer"), "the program panel opens")
        check((page.inner_text("#stagecode") or "").strip().startswith("import cadquery"), "it shows the program",
              page.inner_text("#stagecode")[:40])
        page.click("#backdrop", position={"x": 20, "y": 20})
        page.wait_for_timeout(400)
        check(not page.is_visible("#progdrawer"), "clicking outside closes it")
        page.click("#progBtn")
        page.wait_for_timeout(400)
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        check(not page.is_visible("#progdrawer") and not page.is_visible("#backdrop"), "Escape closes it")

        print("\nparts drawer")
        page.click("#partsBtn")
        page.wait_for_timeout(600)
        check(page.is_visible("#drawer"), "drawer opens")
        parts = page.evaluate("fetch('/api/parts').then(r => r.json())")
        rows = page.query_selector_all("#partlist .row")
        check(len(rows) == min(len(parts), 400), "drawer lists the parts", f"{len(rows)} rows of {len(parts)}")
        dots = page.query_selector_all("#partlist .row:first-child .dot")
        check(len(dots) == len(parts[0]["results"]), "each row shows one mark per model", f"{len(dots)} dots")
        page.click('#filters [data-f="wins"]')
        page.wait_for_timeout(500)
        expect = sum(1 for x in parts if x["ours_correct"] and x["frontier_correct"] == 0)
        check(len(page.query_selector_all("#partlist .row")) == expect, "the 'only we get right' filter matches the data", f"expected {expect}")
        page.fill("#search", "hex")
        page.wait_for_timeout(400)
        titles = [r.inner_text().lower() for r in page.query_selector_all("#partlist .row")]
        check(all("hex" in t for t in titles) if titles else True, "search filters the list", str(titles[:2]))
        page.fill("#search", "")
        page.click('#filters [data-f="all"]')
        page.wait_for_timeout(400)

        total = len(page.query_selector_all("#parts option"))
        page.query_selector_all("#partlist [data-cb]")[0].click()
        page.wait_for_timeout(500)
        check(len(page.query_selector_all("#parts option")) == total - 1, "unticking a part removes it from the picker")
        page.query_selector_all("#partlist [data-cb]")[0].click()
        page.wait_for_timeout(500)
        check(len(page.query_selector_all("#parts option")) == total, "ticking it puts it back")
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        check(not page.is_visible("#drawer"), "Escape closes the drawer")

        print("\nbenchmark table")
        sb = page.evaluate("fetch('/api/scoreboard').then(r => r.json())")
        rows = page.query_selector_all("#scoreboard tbody tr")
        check(len(rows) == len(sb["lanes"]), "one row per lane", f"{len(rows)} rows, {len(sb['lanes'])} lanes")
        for row, lane in zip(rows, sb["lanes"]):
            cells = [c.inner_text() for c in row.query_selector_all("td")]
            check(lane["label"].split(" (")[0] in cells[0], "row label", cells[0])
            check(abs((num(cells[1]) or 0) - 100 * lane["success"]) < 0.11, f"correct % for {lane['label']}", cells[1])
            if lane.get("cost_per_1k_usd") is not None:
                check(abs((num(cells[7]) or 0) - lane["cost_per_1k_usd"]) < 0.015, f"cost for {lane['label']}", cells[7])
            if lane.get("latency_p50") is not None:
                check(abs((num(cells[6]) or 0) - lane["latency_p50"]) < 0.06, f"latency for {lane['label']}", cells[6])

        print("\nrace tab")
        page.click("#tab-race")
        page.wait_for_timeout(1200)
        check(page.is_visible("#lanes"), "race panel visible")
        lanes_cfg = page.evaluate("fetch('/api/config').then(r => r.json())")["lanes"]
        check(len(page.query_selector_all("#lanes > .card")) == len(lanes_cfg), "one card per race lane")
        check(page.eval_on_selector("#examples", "e => !!e.value"), "a part is preselected for the race")
        check(page.eval_on_selector("#rsheet", "i => !i.hidden && i.naturalWidth > 0"), "race sheet image loads")
        page.select_option("#modality", "text")
        page.wait_for_timeout(400)
        check(page.is_visible("#spec") and not page.is_visible("#rsheet"), "written-spec mode swaps the input")
        page.select_option("#modality", "image")
        page.wait_for_timeout(400)

        if args.race:
            print("\nlive race from the compare tab")
            page.click("#tab-compare")
            page.wait_for_timeout(600)
            page.click("#raceThis")
            page.wait_for_timeout(1500)
            check(page.is_visible("#lanes"), "race this part switches tabs and starts")
            page.wait_for_function("document.querySelector('#note').textContent === 'done'", timeout=300_000)
            verdicts = [p.inner_text() for p in page.query_selector_all('#lanes [data-k="pill"]')]
            check(all(("correct" in v or "wrong" in v or "✗" in v) for v in verdicts), "every lane finished", str(verdicts))
            check(canvas_has_content(page, "#target"), "reference solid renders in the race")
            check(any(canvas_has_content(page, f"#lane-{l['key']} .viewer") for l in lanes_cfg), "at least one lane renders its answer")

            print("\nlive race, second run")
            page.click("#go")
            page.wait_for_timeout(2000)
            check(page.eval_on_selector("#go", "b => b.disabled"), "run button disables while racing")
            page.wait_for_function("document.querySelector('#note').textContent === 'done'", timeout=300_000)
            pills = [p.inner_text() for p in page.query_selector_all('#lanes [data-k="pill"]')]
            check(all(("correct" in t or "wrong" in t or "✗" in t) for t in pills), "every lane finishes with a verdict", str(pills))

        page.click("#tab-compare")
        page.wait_for_timeout(800)
        check(canvas_has_content(page, "#stage"), "stage still renders after switching tabs back")

        page.screenshot(path="/tmp/ui_check.png", full_page=True)
        browser.close()

    print("\nconsole/network issues:", len(errors))
    for e in errors[:10]:
        print("   ", e[:160])
    if errors:
        PROBLEMS.append(f"{len(errors)} console/network issues")
    print(f"\n{'PROBLEMS: ' + str(len(PROBLEMS)) if PROBLEMS else 'all checks passed'}")
    sys.exit(1 if PROBLEMS else 0)


if __name__ == "__main__":
    main()
