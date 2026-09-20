
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (x, d = 1) => (x == null ? "–" : Number(x).toFixed(d));
const pct = (x) => (x == null ? "–" : (100 * x).toFixed(1) + "%");
const money = (x) => (x == null ? "–" : "$" + (x < 0.01 ? x.toFixed(4) : x.toFixed(3)));
const loader = new STLLoader();
// The reference is red against the green prediction: ghosted over it, whatever is only red is volume the model
// missed and whatever is only green is volume it invented, which is what the IoU is counting.
const OURS = 0x10b85a, TRUTH = 0xE5342A, RIVALS = [0x60758a, 0x93a08f, 0x7d8fa8];

class Viewer {
  // zoom = margin left around the part. sweep = which percentile of the turn has to fit, 1 = every angle stays
  // inside the frame (and a long part therefore sits small), 0.75 = the widest quarter of the turn may crop.
  constructor(el, color, zoom = 1.12, sweep = 0.75) {
    this.el = el; this.color = color; this.zoom = zoom; this.sweep = sweep; this.part = null; this.target = null;
    this.partSeq = 0; this.targetSeq = 0;  // a mesh that arrives after a newer load started is dropped, not added
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (err) {  // no WebGL: scores and code still work
      this.renderer = null;
      el.insertAdjacentHTML("afterbegin", '<div class="muted" style="padding:10px">3D preview needs WebGL.</div>');
      return;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.prepend(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.001, 1000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.9;
    // rotateSpeed is set from the element's shape in resize(): OrbitControls scales a drag by the canvas HEIGHT in
    // both directions, so on a wide short stage the default spins the part most of a turn for a short sideways drag.
    this.controls.rotateSpeed = 0.5;
    this.controls.enablePan = false;  // nothing here needs panning, and an accidental one loses the part off-frame
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xd8e4d4, 2.0));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(2, 3, 4);
    this.scene.add(key);
    this.group = new THREE.Group();
    this.scene.add(this.group);
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => { this.controls.update(); this.renderer.render(this.scene, this.camera); requestAnimationFrame(loop); };
    loop();
  }
  resize() {
    if (!this.renderer) return;
    const w = this.el.clientWidth || 300, h = this.el.clientHeight || 210;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Keep the feel of a drag the same whatever shape the stage is: OrbitControls divides by height either way, so
    // without this a layout change that shortens the stage silently makes it twitchier. 1.5 * h/w puts a drag across
    // half the width at roughly a quarter turn, which is what it was tuned to.
    this.controls.rotateSpeed = Math.min(1.2, Math.max(0.25, (1.5 * h) / w));
    this.camera.updateProjectionMatrix();
    this.fit();  // the framing depends on the aspect ratio
  }
  async geometry(key) {
    const g = loader.parse(await (await fetch(`/api/mesh/${key}`)).arrayBuffer());
    g.computeVertexNormals();
    return g;
  }
  drop(o) { if (o) { this.group.remove(o); o.geometry.dispose(); } }
  async setPart(key, color) {
    if (!this.renderer) return;
    const seq = ++this.partSeq;
    const g = await this.geometry(key);
    if (seq !== this.partSeq) { g.dispose(); return; }
    this.drop(this.part);
    this.part = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: color ?? this.color, metalness: 0.05, roughness: 0.55, side: THREE.DoubleSide }));
    this.group.add(this.part);
    this.fit();
  }
  /* mode: "ghost" (translucent reference over the prediction), "edges", "solid", or null to remove */
  async setTarget(key, mode = "edges") {
    if (!this.renderer) return;
    const seq = ++this.targetSeq;
    if (!key || !mode) { this.drop(this.target); this.target = null; this.fit(); return; }
    const g = await this.geometry(key);
    if (seq !== this.targetSeq) { g.dispose(); return; }
    this.drop(this.target); this.target = null;
    if (mode === "edges") {
      this.target = new THREE.LineSegments(new THREE.EdgesGeometry(g, 20), new THREE.LineBasicMaterial({ color: 0x51695a, transparent: true, opacity: 0.6 }));
      g.dispose();  // the edge geometry is its own; the solid one is not shown
    } else {
      this.target = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
        color: TRUTH, metalness: 0.05, roughness: 0.65, transparent: mode === "ghost", opacity: mode === "ghost" ? 0.46 : 1,
        depthWrite: mode !== "ghost", side: THREE.DoubleSide,
      }));
    }
    this.group.add(this.target);
    this.fit();
  }
  spin(on) { if (this.controls) this.controls.autoRotate = on; }
  clear() { if (!this.renderer) return; this.partSeq++; this.targetSeq++; this.drop(this.part); this.drop(this.target); this.part = this.target = null; }
  /* Frame the part. Two things make this more than "fit the bounding sphere": the canvas is much wider than it is
     tall, and the stage rotates, so a framing that suits the angle we start at clips the part half a turn later.
     So: stand the camera on the part's shortest axis (a long part then runs across the width, the way the sheet's
     isometric view shows it), and fit the corners at a full turn of azimuths, which is the cylinder the part sweeps
     out. Rotation-safe like the bounding sphere, but as tight as the part's real shape allows. */
  fit() {
    if (!this.renderer) return;
    this.group.quaternion.identity();
    this.group.updateMatrixWorld(true);
    let box = new THREE.Box3().setFromObject(this.group);
    if (box.isEmpty()) return;
    const dir = new THREE.Vector3(1, 0.75, 1.15).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    // These parts are normalised to a 0.75 long axis, so many are thin bars. On a canvas wider than it is tall, stand
    // the part on its shortest axis: the long one then runs across the width, the way the sheet's isometric view shows
    // it, instead of down a narrow strip. Turn the part, never the camera -- OrbitControls fixes its orbit axis from
    // camera.up when it is constructed and never re-reads it, so a camera rolled here would drag against its own axis.
    const size = box.getSize(new THREE.Vector3()).toArray();
    if (this.camera.aspect > 1.3 && Math.max(...size) > 2 * Math.min(...size)) {
      const shortest = [0, 1, 2].sort((a, b) => size[a] - size[b])[0];
      this.group.quaternion.setFromUnitVectors(new THREE.Vector3(+(shortest === 0), +(shortest === 1), +(shortest === 2)), up);
      this.group.updateMatrixWorld(true);
      box = new THREE.Box3().setFromObject(this.group);
    }
    const box3 = [[box.min.x, box.max.x], [box.min.y, box.max.y], [box.min.z, box.max.z]];
    const centre = box.getCenter(new THREE.Vector3());
    const box8 = [];
    for (const x of box3[0]) for (const y of box3[1]) for (const z of box3[2]) box8.push(new THREE.Vector3(x, y, z));

    // The camera orbits `up` through the target, so the target must sit on the part's own axis: centre it along `up`
    // only. Shifting it sideways would tilt the sweep and let a corner swing out of frame.
    const heights = box8.map((v) => v.dot(up));
    const target = centre.clone().addScaledVector(up, (Math.min(...heights) + Math.max(...heights)) / 2 - centre.dot(up));
    // The bounding box at each step of the turn, kept per step rather than pooled. Framing the whole sweep at once
    // means framing a cylinder as wide as the part is long, which leaves a thin bar tiny at every angle. Framing a
    // high percentile instead fills the view for most of the turn and lets only the widest angles crop a little.
    const steps = [];
    for (let k = 0; k < 24; k++) {
      steps.push(box8.map((v) => v.clone().sub(target).applyAxisAngle(up, (k * Math.PI) / 12).add(target)));
    }

    let d = box.getSize(new THREE.Vector3()).length() || 1;
    for (let pass = 0; pass < 6; pass++) {
      this.camera.position.copy(target).addScaledVector(dir, d);
      this.camera.near = d / 200; this.camera.far = d * 200;
      this.camera.updateProjectionMatrix();
      this.camera.lookAt(target);
      this.camera.updateMatrixWorld();
      const perStep = steps.map((pts) => {
        let r = 0;
        for (const v of pts) {
          const q = v.clone().project(this.camera);
          r = Math.max(r, Math.abs(q.x), Math.abs(q.y));
        }
        return r;
      }).sort((a, b) => a - b);
      const reach = perStep[Math.floor(perStep.length * this.sweep)];
      if (!Number.isFinite(reach) || reach <= 0) break;
      d *= reach * this.zoom;
    }
    this.controls.target.copy(target);
    this.controls.minDistance = d * 0.4;  // keep the wheel from diving inside the solid or losing it in the distance
    this.controls.maxDistance = d * 2.5;
    this.camera.position.copy(target).addScaledVector(dir, d);
    this.camera.near = d / 200; this.camera.far = d * 200;
    this.camera.updateProjectionMatrix();
  }
}

/* ---------- tabs ---------- */
function showTab(name) {
  for (const other of ["compare", "race"]) {
    $("#" + other).hidden = other !== name;
    $("#tab-" + other).classList.toggle("on", other === name);
  }
  [stage, targetViewer, ...Object.values(viewers)].forEach((v) => v && v.resize());
}
for (const name of ["compare", "race"]) $("#tab-" + name).onclick = () => showTab(name);

/* ---------- compare: one stage; a dropdown picks whose answer it shows, a drawer curates the parts ---------- */
let PARTS = [], DATA = null, picked = 1, filter = "all";
const EXCLUDED = new Set(JSON.parse(localStorage.getItem("cadabra.excluded") || "[]"));
const stage = new Viewer($("#stage"), OURS, 1.14, 0.92);
// what the stage is actually showing, so scripts/ui_check.py can catch a mesh left behind by a cancelled load
// and check that the reference really does sit on top of the prediction
window.__fit = () => {
  const box = new THREE.Box3().setFromObject(stage.group);
  let lo = [9, 9], hi = [-9, -9];
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
    const q = new THREE.Vector3(x, y, z).project(stage.camera);
    lo = [Math.min(lo[0], q.x), Math.min(lo[1], q.y)]; hi = [Math.max(hi[0], q.x), Math.max(hi[1], q.y)];
  }
  const rel = stage.camera.position.clone().sub(stage.controls.target);
  return { ndc: [lo, hi].map((a) => a.map((v) => +v.toFixed(3))), up: stage.camera.up.toArray(),
           aspect: +stage.camera.aspect.toFixed(3), zoom: stage.zoom,
           radius: +rel.length().toFixed(4), elevation: +Math.asin(rel.y / rel.length()).toFixed(4),
           azimuth: +Math.atan2(rel.x, rel.z).toFixed(4) };
};
window.__stage = () => stage.group.children.map((o) => {
  const a = o.geometry.attributes.position.array;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < a.length; i += 3) for (let j = 0; j < 3; j++) { lo[j] = Math.min(lo[j], a[i + j]); hi[j] = Math.max(hi[j], a[i + j]); }
  return { type: o.type, lo: lo.map((v) => +v.toFixed(4)), hi: hi.map((v) => +v.toFixed(4)),
           at: [o.position.x, o.position.y, o.position.z], opacity: o.material.opacity };
});

const included = () => PARTS.filter((p) => !EXCLUDED.has(p.id));
const saveExcluded = () => localStorage.setItem("cadabra.excluded", JSON.stringify([...EXCLUDED]));

async function loadParts() {
  PARTS = await (await fetch("/api/parts")).json();
  fillParts();
  const wanted = new URLSearchParams(location.search).get("part");
  if (PARTS.length) await showPart(wanted && PARTS.some((p) => p.id === wanted) ? wanted : pickDefault());
}

function fillParts() {
  const list = included();
  const groups = {};
  for (const p of list) (groups[p.tier] ||= []).push(p);
  $("#parts").innerHTML = Object.entries(groups).map(([tier, ps]) =>
    `<optgroup label="${esc(tier)} (${ps.length})">` +
    ps.map((p) => `<option value="${esc(p.id)}">${esc(p.title)} · ${p.n_faces} faces${p.n_parts > 1 ? ", " + p.n_parts + " parts" : ""}</option>`).join("") +
    `</optgroup>`).join("");
  $("#partsCount").textContent = `${list.length} of ${PARTS.length}`;
}

// Open on a part that makes the point: not simple, we get it right, both frontier models do not. Among those prefer
// a compact one, because a part twenty times longer than it is thick renders as a sliver in a wide canvas, then the
// one with the most faces.
const slenderness = (p) => (p.extents && p.extents.length === 3 ? Math.max(...p.extents) / Math.max(Math.min(...p.extents), 1e-6) : 99);
function pickDefault() {
  const pool = included().length ? included() : PARTS;
  const hero = pool.filter((p) => p.ours_correct && p.frontier_correct === 0 && p.tier !== "simple");
  const compact = hero.filter((p) => slenderness(p) <= 5);
  return ((compact.length ? compact : hero).sort((a, b) => (b.n_faces || 0) - (a.n_faces || 0))[0] || pool[0]).id;
}

const entries = () => [{ ...DATA.reference, truth: true, label: "Reference (dataset)" }, ...DATA.lanes];

/* --- model dropdown --- */
function markFor(c) {
  return c.truth ? '<span class="mark truth">◆</span>' : `<span class="mark ${c.success ? "ok" : "no"}">${c.success ? "✓" : "✗"}</span>`;
}
function drawModelMenu() {
  const list = entries();
  const c = list[picked];
  $("#modelBtnText").innerHTML = `${markFor(c)} ${esc(c.label)}` +
    (c.truth ? "" : ` <span class="iou" style="margin-left:8px">${c.iou_aligned == null ? "no solid" : "IoU " + c.iou_aligned.toFixed(3)}</span>`);
  $("#modelMenu").innerHTML = list.map((e, i) =>
    `<button data-i="${i}" class="${i === picked ? "on" : ""}">${markFor(e)} <span>${esc(e.label)}</span>
      <span class="iou">${e.truth ? "ground truth" : e.iou_aligned == null ? "no solid" : "IoU " + e.iou_aligned.toFixed(3)}</span></button>`).join("");
  $("#modelMenu").querySelectorAll("[data-i]").forEach((b) => (b.onclick = () => { closeMenu(); select(+b.dataset.i); }));
}
const closeMenu = () => ($("#modelMenu").hidden = true);
$("#modelBtn").onclick = (e) => { e.stopPropagation(); $("#modelMenu").hidden = !$("#modelMenu").hidden; };
document.addEventListener("click", (e) => { if (!e.target.closest(".dropdown")) closeMenu(); });

async function select(i) {
  picked = i;
  const c = entries()[i];
  drawModelMenu();
  stage.clear();
  const overlay = $("#overlay").checked;
  $("#stagehint").textContent = c.truth ? "the dataset's reference solid"
    : overlay ? "prediction solid, reference ghosted over it in red" : "prediction only";
  $("#stagestats").innerHTML = c.truth
    ? `${c.stats ? c.stats.n_faces + " faces · what every model is scored against" : ""}`
    : `${c.success ? '<b style="color:#06682F">correct</b>' : '<b style="color:#B4482C">wrong</b>'} · IoU ${c.iou_aligned == null ? "–" : c.iou_aligned.toFixed(3)}` +
      ` · ${num(c.e2e_s, 1)} s · ${money(c.cost_usd)} · ${c.output_tokens ?? "–"} output tokens` +
      (c.mesh ? "" : " · " + esc((c.error || "did not build").slice(0, 70)));
  $("#codeowner").textContent = `program · ${c.truth ? "dataset reference" : c.label}`;
  $("#stagecode").textContent = c.code || "";
  if (c.truth) { if (c.mesh) await stage.setPart(c.mesh, TRUTH); return; }
  if (c.mesh) await stage.setPart(c.mesh, c.ours ? OURS : RIVALS[0]);
  if (overlay && DATA.reference.mesh) await stage.setTarget(DATA.reference.mesh, c.mesh ? "ghost" : "solid");
}

async function showPart(id) {
  $("#parts").value = id;
  $("#partnote").textContent = "building the solids…";
  let res;
  try {
    res = await fetch(`/api/compare/${encodeURIComponent(id)}`);
    if (!res.ok) throw new Error(await res.text());
    DATA = await res.json();
  } catch (err) {
    $("#partnote").textContent = "could not load this part: " + String(err).slice(0, 80);
    return;
  }
  $("#sheet").src = DATA.sheet;
  const b = DATA.bbox;
  $("#parttitle").textContent = DATA.title;
  $("#bboxline").textContent = `${DATA.tier_faces} faces${DATA.n_parts > 1 ? " · " + DATA.n_parts + " parts" : ""}` +
    (b ? ` · bounding box ${b[0].toFixed(4)} × ${b[1].toFixed(4)} × ${b[2].toFixed(4)}` : "");
  $("#partnote").textContent = "held out from training · answers replayed from the benchmark runs";
  const want = +(new URLSearchParams(location.search).get("model") ?? 1);
  await select(Math.max(0, Math.min(Number.isFinite(want) ? want : 1, entries().length - 1)));
  if (!$("#drawer").hidden) drawList();
}

function step(delta) {
  const ids = [...$("#parts").options].map((o) => o.value);
  if (!ids.length) return;
  const i = ids.indexOf($("#parts").value);
  showPart(ids[(i + delta + ids.length) % ids.length]);
}
$("#prev").onclick = () => step(-1);
$("#next").onclick = () => step(1);
$("#parts").onchange = (e) => showPart(e.target.value);
$("#overlay").onchange = () => select(picked);
$("#spin").onchange = (e) => stage.spin(e.target.checked);
$("#raceThis").onclick = () => {
  if (!DATA) return;
  useExample({ id: DATA.id, spec: DATA.spec, sheet: DATA.sheet, bbox: DATA.bbox, title: DATA.title });
  showTab("race");
  $("#go").click();
};

/* --- parts drawer: include or exclude parts, and see who got each one right --- */
function shown() {
  const q = $("#search").value.trim().toLowerCase();
  return PARTS.filter((p) => {
    if (q && !(p.title.toLowerCase().includes(q) || p.id.toLowerCase().includes(q))) return false;
    if (filter === "wins") return p.ours_correct && p.frontier_correct === 0;
    if (filter === "missed") return !p.ours_correct;
    if (filter === "hard") return p.tier === "medium" || p.tier === "complex" || p.tier === "multi-part";
    return true;
  });
}

const shortLabel = (label) => label
  .replace("Cadabra 4B (ours)", "ours").replace(" · best of ", " ×").replace(" · 1 sample", "")
  .replace("GLM-5.3 ", "").replace("Kimi ", "").replace(" (high)", "");

function drawList() {
  const list = shown();
  const models = (PARTS.find((p) => p.results)?.results) || [];
  $("#legend").innerHTML = "marks: " + models.map((r) =>
    `<span class="dot na" style="margin:0 5px 0 0"></span>${esc(shortLabel(r.label))}`).join(" &nbsp; ");
  $("#drawerCount").textContent = `${list.length} shown · ${included().length} included`;
  $("#partlist").innerHTML = list.slice(0, 400).map((p) => `
    <div class="row ${p.id === $("#parts").value ? "on" : ""}" data-id="${esc(p.id)}">
      <input type="checkbox" ${EXCLUDED.has(p.id) ? "" : "checked"} data-cb="${esc(p.id)}" title="include in the part list">
      <div style="min-width:0">
        <div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(p.title)}</div>
        <div class="meta">${p.tier} · ${p.n_faces} faces${p.n_parts > 1 ? " · " + p.n_parts + " parts" : ""}</div>
      </div>
      <div class="marks">${(p.results || []).map((r) => `<span class="dot ${r.success ? "ok" : "no"}" title="${esc(r.label)}: ${r.success ? "correct" : "wrong"}${r.iou == null ? "" : " (IoU " + r.iou.toFixed(3) + ")"}"></span>`).join("")}</div>
    </div>`).join("") + (list.length > 400 ? `<div class="label" style="padding:8px 10px">showing the first 400</div>` : "");
  $("#partlist").querySelectorAll("[data-cb]").forEach((cb) => (cb.onclick = (e) => {
    e.stopPropagation();
    const id = cb.dataset.cb;
    cb.checked ? EXCLUDED.delete(id) : EXCLUDED.add(id);
    saveExcluded(); fillParts(); drawList();
  }));
  $("#partlist").querySelectorAll(".row").forEach((row) => (row.onclick = () => showPart(row.dataset.id)));
}

function openDrawer(open) {
  $("#drawer").hidden = !open;
  if (open) $("#progdrawer").hidden = true;
  syncBackdrop();
  if (open) { drawList(); $("#search").focus(); }
}
function openProgram(open) {
  $("#progdrawer").hidden = !open;
  if (open) $("#drawer").hidden = true;
  syncBackdrop();
}
const syncBackdrop = () => ($("#backdrop").hidden = $("#drawer").hidden && $("#progdrawer").hidden);
const closePanels = () => { openDrawer(false); openProgram(false); };
$("#partsBtn").onclick = () => openDrawer($("#drawer").hidden);
$("#progBtn").onclick = () => openProgram($("#progdrawer").hidden);
$("#drawerClose").onclick = () => openDrawer(false);
$("#progClose").onclick = () => openProgram(false);
$("#backdrop").onclick = closePanels;
$("#search").oninput = drawList;
$("#filters").querySelectorAll("[data-f]").forEach((b) => (b.onclick = () => {
  filter = b.dataset.f;
  $("#filters").querySelectorAll("[data-f]").forEach((x) => x.classList.toggle("on", x === b));
  drawList();
}));
$("#selAll").onclick = () => { shown().forEach((p) => EXCLUDED.delete(p.id)); saveExcluded(); fillParts(); drawList(); };
$("#selNone").onclick = () => {
  shown().forEach((p) => EXCLUDED.add(p.id));
  if (!included().length) EXCLUDED.delete(shown()[0]?.id);  // never empty the list
  saveExcluded(); fillParts(); drawList();
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closePanels(); closeMenu(); }
  if ($("#compare").hidden || !$("#drawer").hidden || !$("#progdrawer").hidden || e.target.matches("input, select, textarea")) return;
  if (e.key === "ArrowLeft") step(-1);
  if (e.key === "ArrowRight") step(1);
});

/* ---------- live race ---------- */
let CONFIG = null, EXAMPLES = [], currentExample = null;
const viewers = {};
const targetViewer = new Viewer($("#target"), TRUTH);
function card(key) { return document.getElementById("lane-" + key); }
function setv(key, field, value) { const el = card(key)?.querySelector(`[data-k="${field}"]`); if (el) el.textContent = value; }
function pill(key, cls, text) { const el = card(key)?.querySelector('[data-k="pill"]'); if (el) { el.className = "pill " + cls; el.textContent = text; } }

function buildLanes(laneList) {
  $("#lanes").innerHTML = "";
  for (const k in viewers) delete viewers[k];
  let i = 0;
  for (const l of laneList) {
    const el = document.createElement("div");
    el.className = "card" + (l.dedicated ? " ours" : "");
    el.id = "lane-" + l.key;
    el.innerHTML = `
      <div class="head"><h3>${esc(l.label)}</h3><span class="pill">${l.dedicated ? "ours · Baseten" : "Model API" + (l.effort ? " · " + l.effort : "")}</span></div>
      <div class="slug">${esc(l.model)}</div>
      <div class="meters">
        <div class="m"><b data-k="ttft">–</b><span>first token s</span></div>
        <div class="m"><b data-k="e2e">–</b><span>total s</span></div>
        <div class="m"><b data-k="tok">–</b><span>tokens</span></div>
        <div class="m"><b data-k="cost">–</b><span>${l.dedicated ? "$/part (GPU)" : "$/part"}</span></div>
      </div>
      <div class="stats" data-k="status">idle</div>
      <div class="viewer" data-k="viewer"><span class="hint">prediction over reference (edges)</span></div>
      <div><span class="pill" data-k="pill">waiting</span></div>
      <details><summary>code</summary><pre data-k="code"></pre></details>`;
    $("#lanes").appendChild(el);
    viewers[l.key] = new Viewer(el.querySelector('[data-k="viewer"]'), l.dedicated ? OURS : RIVALS[i++ % RIVALS.length]);
  }
}

function showInput() {
  const image = $("#modality").value === "image";
  $("#spec").hidden = image;
  $("#sheetbox").hidden = !image;
  $("#note").textContent = image
    ? "Every model gets the same sheet and bounding box. Parts are held out."
    : "Every model gets the same written spec.";
  $("#rsheet").hidden = !(image && currentExample && currentExample.sheet);
  if (image && currentExample && currentExample.sheet) {
    $("#rsheet").src = currentExample.sheet;
    const b = currentExample.bbox;
    $("#rbbox").textContent = b ? `bounding box X ${b[0].toFixed(4)}, Y ${b[1].toFixed(4)}, Z ${b[2].toFixed(4)}` : "";
  }
}
$("#modality").onchange = showInput;
function useExample(x) {
  currentExample = x;
  if (![...$("#examples").options].some((o) => o.value === x.id)) {
    $("#examples").add(new Option(x.title || x.id, x.id));
    EXAMPLES.push(x);
  }
  $("#examples").value = x.id;
  if (x.spec) $("#spec").value = x.spec;
  showInput();
}
$("#examples").onchange = (e) => {
  const x = EXAMPLES.find((y) => y.id === e.target.value);
  if (x) useExample(x);
};

async function handle(ev, state) {
  if (ev.type === "target") {
    await targetViewer.setTarget(ev.mesh, "solid");
    $("#tstats").textContent = `· ${ev.stats.n_faces} faces`;
    for (const k in viewers) viewers[k].setTarget(ev.mesh, "edges");
    return;
  }
  const s = state[ev.lane];
  if (!s) return;
  if ((ev.type === "reasoning" || ev.type === "content") && s.ttft == null) {
    s.ttft = (performance.now() - s.t0) / 1000;
    setv(ev.lane, "ttft", s.ttft.toFixed(2));
  }
  if (ev.type === "reasoning") {
    s.reasoning += ev.text.length;
    setv(ev.lane, "status", `thinking… (~${Math.round(s.reasoning / 4)} tokens)`);
  } else if (ev.type === "content") {
    s.code += ev.text;
    setv(ev.lane, "status", "writing code…");
    setv(ev.lane, "code", s.code);
  } else if (ev.type === "status") {
    setv(ev.lane, "status", ev.text);
  } else if (ev.type === "restart") {
    s.code = ""; s.reasoning = 0;
    setv(ev.lane, "status", "connection dropped, retrying…");
  } else if (ev.type === "generated") {
    s.done = true;
    setv(ev.lane, "e2e", num(ev.metrics.e2e_s, 1));
    setv(ev.lane, "tok", ev.metrics.output_tokens ?? "–");
    setv(ev.lane, "cost", money(ev.metrics.cost_usd));
    setv(ev.lane, "status", ev.error ? "API error: " + ev.error : "building the solid…");
  } else if (ev.type === "graded") {
    s.done = true;
    if (ev.code) setv(ev.lane, "code", ev.code);
    if (!ev.runs) {
      pill(ev.lane, "no", "✗ " + (ev.error || "failed").slice(0, 60));
      setv(ev.lane, "status", "code did not produce a solid");
      return;
    }
    await viewers[ev.lane].setPart(ev.mesh);
    if (ev.iou_aligned == null) pill(ev.lane, "", `built · ${ev.stats?.n_faces ?? "?"} faces`);
    else if (ev.iou_aligned >= 0.9) pill(ev.lane, "ok", `✓ correct · IoU ${ev.iou_aligned.toFixed(3)}`);
    else pill(ev.lane, "no", `✗ wrong part · IoU ${ev.iou_aligned.toFixed(3)}`);
    setv(ev.lane, "status", ev.candidates ? `picked from ${ev.candidates} by render-check (${(ev.render_score ?? 0).toFixed(3)})` : "done");
  } else if (ev.type === "error") {
    s.done = true;
    pill(ev.lane, "no", "✗ " + ev.error.slice(0, 60));
  }
}

/* The published build has no server to race against, so it replays this part's answers from the benchmark run.
   Every figure on screen is the measured one. Only the waiting is compressed, by a single factor applied to all
   lanes, so a 49 second lane does not stall the page; the timers still count in recorded seconds. */
let replaySpeed = 1;
async function replayRace(state) {
  const id = currentExample && currentExample.id;
  if (!id) { $("#note").textContent = "Pick a part first."; return; }
  const d = await (await fetch(`/api/compare/${encodeURIComponent(id)}`)).json();
  if (d.reference && d.reference.mesh) await handle({ type: "target", mesh: d.reference.mesh, stats: d.reference.stats }, state);
  const slowest = Math.max(1, ...d.lanes.map((l) => l.e2e_s || 0));
  replaySpeed = Math.min(1, 9 / slowest);  // the slowest lane lands in about nine seconds
  await Promise.all(d.lanes.map(async (l) => {
    if (!state[l.key]) return;
    await new Promise((done) => setTimeout(done, (l.e2e_s || 0) * 1000 * replaySpeed));
    await handle({ type: "generated", lane: l.key,
                   metrics: { e2e_s: l.e2e_s, output_tokens: l.output_tokens, cost_usd: l.cost_usd } }, state);
    await handle({ type: "graded", lane: l.key, code: l.code, runs: l.runs, error: l.error, mesh: l.mesh,
                   iou_aligned: l.iou_aligned, stats: l.stats, candidates: l.best_of > 1 ? l.best_of : null }, state);
  }));
  replaySpeed = 1;
}

$("#go").onclick = async () => {
  const spec = $("#spec").value.trim();
  const image = $("#modality").value === "image";
  if (image && !(currentExample && currentExample.sheet)) { $("#note").textContent = "Pick a held-out part first."; return; }
  if (!image && !spec) { $("#note").textContent = "Write a spec first."; return; }
  const state = {};
  targetViewer.clear();
  $("#tstats").textContent = "";
  for (const l of CONFIG.lanes) {
    state[l.key] = { t0: performance.now(), done: false, reasoning: 0, ttft: null, code: "" };
    for (const k of ["ttft", "e2e", "tok", "cost"]) setv(l.key, k, "–");
    setv(l.key, "status", "waiting for first token…");
    setv(l.key, "code", "");
    pill(l.key, "", "running");
    viewers[l.key].clear();
  }
  $("#go").disabled = true;
  $("#note").textContent = "racing…";
  const clock = setInterval(() => {
    for (const k in state) if (!state[k].done) setv(k, "e2e", ((performance.now() - state[k].t0) / 1000 / replaySpeed).toFixed(1));
  }, 100);
  try {
    if (window.CADABRA_STATIC) {
      await replayRace(state);
      $("#note").textContent = "done";
      return;
    }
    const res = await fetch("/api/race", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, example_id: currentExample ? currentExample.id : null, modality: $("#modality").value }),
    });
    if (!res.ok) { $("#note").textContent = "race failed: " + (await res.text()).slice(0, 120); return; }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (chunk.startsWith("data: ")) await handle(JSON.parse(chunk.slice(6)), state);
      }
    }
    $("#note").textContent = "done";
  } finally {
    clearInterval(clock);
    $("#go").disabled = false;
  }
};

/* ---------- benchmark table ---------- */
async function loadScoreboard() {
  const sb = await (await fetch("/api/scoreboard")).json();
  if (!sb || !sb.lanes) return;
  $("#sbmeta").textContent = `· ${sb.n} held-out parts, never seen in training`;
  const runs = [...new Set(sb.lanes.flatMap((l) => l.runs || []))];
  $("#sbfoot").textContent = `Success = the program runs and its solid overlaps the reference by 0.9 or more (aligned IoU). `
    + `Cost is API list price for the frontier lanes and the GPU hourly price over measured throughput for ours. `
    + `Runs: ${runs.join(", ")}.`;
  const tier = (l, k) => (l.tiers && l.tiers[k] ? pct(l.tiers[k].success) : "–");
  $("#scoreboard").innerHTML = `<table><thead><tr>
      <th>Model</th><th>Correct [95% CI]</th><th>Simple</th><th>Medium</th><th>Complex</th><th>Multi-part</th>
      <th>Latency p50</th><th>$ / 1K parts</th></tr></thead><tbody>` +
    sb.lanes.map((l) => `<tr class="${l.lane === "specialist" ? "ours" : ""}">
      <td>${esc(l.label)}${l.best_of > 1 ? " · best of " + l.best_of : ""}</td>
      <td>${pct(l.success)} <span class="muted">[${pct(l.success_ci95[0])}, ${pct(l.success_ci95[1])}]</span></td>
      <td>${tier(l, "simple (<=6 faces)")}</td><td>${tier(l, "medium (7-12)")}</td>
      <td>${tier(l, "complex (>=13)")}</td><td>${tier(l, "multi-part")}</td>
      <td>${num(l.latency_p50, 1)} s</td><td>${l.cost_per_1k_usd == null ? "–" : "$" + l.cost_per_1k_usd.toFixed(2)}</td>
    </tr>`).join("") + "</tbody></table>";
}

/* ---------- start ---------- */

/* Called once by the Demo component after React has put the markup in the document.
   Everything above is the page's own code, unchanged: it works on ids, not on React state. */
export async function start() {
  CONFIG = await (await fetch("/api/config")).json();
  $("#modality").value = CONFIG.default_modality || "image";
  buildLanes(CONFIG.lanes);
  EXAMPLES = await (await fetch("/api/examples")).json();
  $("#examples").innerHTML = EXAMPLES.map((e) => `<option value="${esc(e.id)}">${esc(e.title)} · ${e.n_parts}-part, ${e.n_faces} faces</option>`).join("");
  if (EXAMPLES.length) useExample(EXAMPLES[0]); else showInput();
  await loadParts();
  loadScoreboard();
  if (new URLSearchParams(location.search).get("tab") === "race") showTab("race");
}
