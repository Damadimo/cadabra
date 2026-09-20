"use client";

import { useEffect, useRef } from "react";
import "./demo-ids.css";

/* The interactive demo. React owns the markup; the behaviour lives in lib/demo.js, which is the same code the local
   FastAPI app runs and drives the page through ids rather than through React state. Rewriting a three.js viewer,
   an orbit camera and a grading replay as components would buy nothing and lose everything that has been tested. */
export default function Demo() {
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;  // React runs effects twice in dev; the demo must boot once
    started.current = true;

    // The published build is static: there is no API, so serve the page's own calls from the exported files.
    if (process.env.NODE_ENV === "production") {
      window.CADABRA_STATIC = true;
      const real = window.fetch.bind(window);
      window.fetch = (input, init) => {
        const url = typeof input === "string" ? input : input.url;
        const m = /^\/api\/(.*)$/.exec(url || "");
        if (!m) return real(input, init);
        const rest = m[1];
        const file = (p) => real("/data/" + p, init);
        const id = (x) => decodeURIComponent(x).replace(/[:/]/g, "_");
        if (rest === "parts") return file("parts.json");
        if (rest === "scoreboard") return file("scoreboard.json");
        if (rest === "config") return file("config.json");
        if (rest === "examples") return file("examples.json");
        if (rest.startsWith("compare/")) return file("compare/" + id(rest.slice(8)) + ".json");
        if (rest.startsWith("mesh/")) return file("mesh/" + rest.slice(5) + ".stl");
        if (rest.startsWith("sheet/")) return file("sheet/" + id(rest.slice(6)) + ".png");
        return Promise.resolve(new Response("not in the static build", { status: 404 }));
      };
    }

    import("@/lib/demo.js").then((m) => m.start()).catch((e) => {
      console.error(e);
      const note = document.getElementById("partnote");
      if (note) note.textContent = "could not load the demo: " + e.message;
    });
  }, []);

  return (
    <>
      <main>
        <section id="compare">
          <div className="bar">
            <button id="prev" className="icon" title="previous part (←)">←</button>
            <select id="parts"></select>
            <button id="next" className="icon" title="next part (→)">→</button>
            <button id="partsBtn">Parts <span className="iou" id="partsCount" style={{ margin: 0 }}></span></button>
            <button id="raceThis">Race this part live</button>
          </div>

          <div className="split flush" style={{ marginTop: 22 }}>
            <div className="card sheet" id="sheetcard">
              <h3 style={{ margin: "0 0 5px" }} id="parttitle"></h3>
              <div className="label" style={{ marginBottom: 16 }} id="bboxline"></div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img id="sheet" alt="drawing sheet" />
              <div className="foot" id="partnote"></div>
            </div>

            <div className="card">
              <div className="bar">
                <div className="dropdown">
                  <button id="modelBtn" className="model-btn"><span id="modelBtnText">choose a model</span><span aria-hidden="true">▾</span></button>
                  <div className="menu" id="modelMenu" hidden></div>
                </div>
                <label className="check"><input type="checkbox" id="overlay" defaultChecked /> overlay reference</label>
                <label className="check"><input type="checkbox" id="spin" defaultChecked /> rotate</label>
                <button id="progBtn" className="icon">Program</button>
              </div>
              <div className="stats" id="stagestats" style={{ marginTop: 14 }}></div>
              <div className="stage" id="stage"><span className="hint" id="stagehint"></span></div>
              <div className="foot">Volumetric overlap with the reference solid, maximised over the 24 axis rotations. Correct at 0.9 or above.</div>
            </div>
          </div>
        </section>

        <section id="race" hidden>
          <div className="bar">
            <select id="examples"></select>
            <select id="modality"><option value="image">Input: drawing sheet</option></select>
            <button id="go" className="primary">Run ▶</button>
            <span className="muted" id="note"></span>
          </div>
          <div className="split" style={{ marginTop: 14 }}>
            <div className="card sheet">
              <div id="sheetbox">
                <div className="label">input · drawing sheet</div>
                <div className="label" style={{ margin: "6px 0 10px" }} id="rbbox"></div>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img id="rsheet" alt="drawing sheet" hidden />
              </div>
              <textarea id="spec" placeholder="Describe a part with exact dimensions…"></textarea>
              <div style={{ marginTop: 10 }}>
                <h3>Target part <span className="muted" id="tstats"></span></h3>
                <div className="viewer" id="target" style={{ marginTop: 6 }}><span className="hint">reference solid</span></div>
              </div>
            </div>
            <div>
              <div className="cards" id="lanes"></div>
              <div className="foot">Frontier lanes call Baseten Model APIs at high reasoning effort with two worked examples; ours gets the
                zero-shot prompt it was trained on and samples 8 programs, keeping the best match to the drawing. $/part for ours is
                GPU seconds at concurrency 1, an upper bound; the table below amortises the GPU over measured throughput.</div>
            </div>
          </div>
        </section>

        <section className="card">
          <h2>Held-out benchmark <span className="muted" id="sbmeta"></span></h2>
          <div className="scroll" id="scoreboard" style={{ marginTop: 10 }}><span className="muted">No benchmark run yet.</span></div>
          <div className="foot" id="sbfoot"></div>
        </section>
      </main>

      <div className="backdrop" id="backdrop" hidden></div>

      <aside className="drawer" id="progdrawer" hidden aria-label="program">
        <div className="bar" style={{ justifyContent: "space-between" }}>
          <h2>Program</h2>
          <button id="progClose" className="icon" title="close (Esc)">✕</button>
        </div>
        <div className="label" id="codeowner" style={{ marginTop: 12 }}>program</div>
        <pre id="stagecode" style={{ marginTop: 8, maxHeight: "none" }}></pre>
      </aside>

      <aside className="drawer" id="drawer" hidden aria-label="parts">
        <div className="bar" style={{ justifyContent: "space-between" }}>
          <h2>Parts</h2>
          <button id="drawerClose" className="icon" title="close (Esc)">✕</button>
        </div>
        <div style={{ marginTop: 12 }}><input type="search" id="search" placeholder="search parts…" /></div>
        <div className="bar filters" style={{ marginTop: 10 }} id="filters">
          <button data-f="all" className="on">All</button>
          <button data-f="wins">Only we get right</button>
          <button data-f="missed">We get wrong</button>
          <button data-f="hard">Medium &amp; complex</button>
        </div>
        <div className="bar" style={{ marginTop: 10 }}>
          <button id="selAll" className="icon">Include shown</button>
          <button id="selNone" className="icon">Exclude shown</button>
          <span className="label" id="drawerCount"></span>
        </div>
        <div className="label" id="legend" style={{ marginTop: 12 }}></div>
        <div id="partlist" style={{ marginTop: 6 }}></div>
      </aside>
    </>
  );
}
