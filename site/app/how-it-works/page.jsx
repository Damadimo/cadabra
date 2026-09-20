/* The written walkthrough: what the model is, how the data was built, how a part is graded, and what the
   numbers are. Static content, so it renders on the server and ships as HTML. */
import "./how-it-works.css";

export const metadata = { title: "Cadabra · How it works" };

export default function HowItWorks() {
  return (
    <>


<header>
  <div className="inner">
    <h1>Cadabra</h1>
    <nav>
      <a href="/">Compare</a>
      <a href="/?tab=race">Live race</a>
      <a href="/how-it-works" className="on">How it works</a>
    </nav>
  </div>
</header>

<div className="shell">

  <nav className="side" aria-label="Sections">
    <ol>
      <li><a href="#pipeline"><i>01</i>Pipeline</a></li>
      <li><a href="#data"><i>02</i>Data</a></li>
      <li><a href="#grading"><i>03</i>Grading</a></li>
      <li><a href="#results"><i>04</i>Results</a></li>
      <li><a href="#leakage"><i>05</i>Leakage</a></li>
      <li><a href="#negative"><i>06</i>Negative result</a></li>
      <li><a href="#provenance"><i>07</i>Provenance</a></li>
    </ol>
  </nav>

<main>

  <section className="hero">
    <h1 className="display">Frontier models write CAD from a spec. Give them a drawing and they fall apart.</h1>
    <p className="lede">
      Kimi K3 gets 95% of our held-out parts right from a text description. Hand it the same part as a four-view
      drawing sheet and it drops to 61.8%. That gap is narrow, it matters, and it can be graded exactly.
    </p>

    <div className="headline">
      <b>80.7%</b>
      <span>correct on 497 held-out parts, sampling eight programs and picking one without an answer key</span>
    </div>
    <div className="against">
      <div><span>GLM-5.3 Flash</span><b>66.2%</b></div>
      <div><span>Kimi K3</span><b>61.8%</b></div>
      <div><span>the same 4B, untrained</span><b>14.7%</b></div>
      <div><span>our cost per 1K parts</span><b>$0.22</b></div>
    </div>
  </section>

  <section id="pipeline">
    <div className="shead">
      <span className="eyebrow">01 · Pipeline</span>
      <h2>From a public dataset to a model that checks its own work</h2>
      <p className="standfirst">Public CAD data is noisy, so every stage either checks the data or removes a way for the result to be wrong.</p>
    </div>

    <ol className="steps">
      <li><span className="n">1</span><div>
        <h3>Audit every reference program</h3>
        <p>Execute all 82,659 programs in CAD-Coder and check each resulting solid against the dimensions its own spec states. This is where we found the data was wrong.</p>
      </div></li>
      <li><span className="n">2</span><div>
        <h3>Build clean splits</h3>
        <p>Hold 500 parts out by source part, then drop every training part whose geometry signature matches a benchmark part. 1,886 were removed this way.</p>
      </div></li>
      <li className="key"><span className="n">3</span><div>
        <h3>Render each part as a drawing</h3>
        <p>Front, top and right at one shared scale, plus an isometric view, rendered from the part's own reference program so the drawing and the answer always agree. 30,260 sheets.</p>
      </div></li>
      <li className="key"><span className="n">4</span><div>
        <h3>LoRA fine-tune Qwen3-VL-4B</h3>
        <p>Rank 64 on Baseten Training, adapters on the language model with the vision tower frozen, loss on the generated code only. Two epochs: 2.9 hours on one H100, then 55 minutes on four.</p>
      </div></li>
      <li><span className="n">5</span><div>
        <h3>Serve from the training checkpoint</h3>
        <p>Merged weights are pulled straight out of the training job and served with vLLM. No copy step between training and serving.</p>
      </div></li>
      <li className="key"><span className="n">6</span><div>
        <h3>Sample, then check its own work</h3>
        <p>Draw eight programs, render each candidate the way the input sheet was drawn, and keep whichever matches the drawing best. The verifier never sees the reference solid.</p>
      </div></li>
      <li><span className="n">7</span><div>
        <h3>Grade by executing the geometry</h3>
        <p>Run the program in a sandbox and compare its solid to the reference by volumetric intersection over union. No model judges another model anywhere in this project.</p>
      </div></li>
    </ol>
  </section>

  <section id="data">
    <div className="shead">
      <span className="eyebrow">02 · Data</span>
      <h2>Look at your data</h2>
      <p className="standfirst">We ran all 82,659 reference programs before training anything, and checked each solid against the size its own specification claimed.</p>
    </div>

    <div className="callout warn">
      <b>14%</b>
      <p>of the official test split contradicts its own specification, on single-part rows where the stated size is
      checkable. The hand-checked <code>train_high</code> split came in at 2.3%. Anyone reporting accuracy against
      that test split is reporting partly on noise.</p>
    </div>

    <div className="cols full" style={{"marginTop": "var(--s7)"}}>
      <div>
        <span className="eyebrow">Inconsistent transforms</span>
        <p>References apply the specs' stated rotations and translations inconsistently, so the metric aligns both solids first, taking the best overlap across the 24 axis rotations.</p>
      </div>
      <div>
        <span className="eyebrow">Still strict</span>
        <p>Alignment forgives placement, not shape or size. A part built 10% too large scores 0.75 and fails, well short of the 0.9 bar.</p>
      </div>
      <div>
        <span className="eyebrow">Code that writes to disk</span>
        <p>Some reference programs export files while being evaluated. The sandbox strips export calls, allows only three imports, and runs in a throwaway directory.</p>
      </div>
    </div>
  </section>

  <section id="grading">
    <div className="shead">
      <span className="eyebrow">03 · Grading</span>
      <h2>How a part is scored</h2>
      <p className="standfirst">A prediction is correct when its program runs and the solid it produces overlaps the reference by 0.9 or more. Everything else is a miss, including code that builds something plausible but wrong.</p>
    </div>

    <div className="pair full">
      <div>
        <span className="eyebrow">The metric</span>
        <p style={{"font": "400 15px/1.6 var(--mono)", "color": "var(--g10)", "marginBottom": "var(--s3)"}}>IoU = volume(A ∩ B) / volume(A ∪ B)</p>
        <p>Exact boolean volumes from OpenCascade, not a mesh approximation and not a sampled estimate. We take the best value over all 24 axis-aligned rotations after centering both solids.</p>
      </div>
      <div>
        <span className="eyebrow">The verifier, which never sees the answer</span>
        <p>To pick between candidate programs at inference time, each one is built and re-rendered from the same three orthographic views as the input sheet. The score is silhouette overlap across those views, multiplied by how closely the candidate's bounding box matches the one stated in the prompt.</p>
      </div>
    </div>

    <div className="callout">
      <b>0.96 AUC</b>
      <p>separating correct programs from incorrect ones across 689 real frontier-model answers spanning 237 parts.
      Given several answers for one part it picks a correct one 98% of the time, against 76% for a random pick. It is
      not specific to our model: the same verifier lifts Kimi K3 from 28% to 48% on hard parts.</p>
    </div>
  </section>

  <section id="results">
    <div className="shead">
      <span className="eyebrow">04 · Results</span>
      <h2>Every lane on the same 497 parts</h2>
      <p className="standfirst">Bootstrap 95% confidence intervals on every row. A request the API never answered is excluded and listed, never counted as a wrong answer.</p>
    </div>

    <div className="scroll full">
      <table>
        <thead>
          <tr><th>Model</th><th>Correct</th><th>95% CI</th><th>Simple</th><th>Medium</th><th>Complex</th><th>Multi</th><th>$ / 1K</th><th>p50</th></tr>
        </thead>
        <tbody>
          <tr className="ours"><td>Cadabra 4B, best of 8</td><td>80.7%</td><td>77–84</td><td>95.6%</td><td>61.3%</td><td>39.0%</td><td>58.1%</td><td>$1.19</td><td>2.3 s</td></tr>
          <tr className="ours"><td>Cadabra 4B, 1 sample</td><td>74.8%</td><td>71–79</td><td>92.5%</td><td>52.1%</td><td>25.4%</td><td>53.5%</td><td>$0.22</td><td>2.0 s</td></tr>
          <tr><td>GLM-5.3 Flash (high)</td><td>66.2%</td><td>62–70</td><td>85.9%</td><td>37.0%</td><td>18.6%</td><td>39.5%</td><td>$1.00</td><td>3.7 s</td></tr>
          <tr><td>Kimi K3 (high)</td><td>61.8%</td><td>58–66</td><td>81.5%</td><td>30.3%</td><td>18.6%</td><td>39.5%</td><td>$36.82</td><td>10.4 s</td></tr>
          <tr><td>Qwen3-VL-4B, untuned</td><td>14.7%</td><td>11–18</td><td>19.4%</td><td>8.4%</td><td>1.7%</td><td>16.3%</td><td>–</td><td>–</td></tr>
        </tbody>
      </table>
    </div>
    <p className="note">Tiers are the reference solid's face count: simple ≤ 6, medium 7 to 12, complex ≥ 13. Frontier lanes get two worked examples and high reasoning effort. Ours gets neither.</p>

    <div className="pair full" style={{"marginTop": "var(--s7)"}}>
      <figure>
        <img src="/chart/results_by_tier.png" alt="Correct answers by part complexity, one bar per model, with 95% confidence intervals" />
        <figcaption>The gap widens with complexity. On parts with 13 or more faces we are at 39.0% against 18.6% for both frontier models.</figcaption>
      </figure>
      <figure>
        <img src="/chart/learning_curve.png" alt="Accuracy against training step, with the frontier models drawn as reference lines" />
        <figcaption>13% untuned, 61% by step 200, 68% by step 1,000, 73.5% after one epoch and 77.0% after two, on the frontier's 200 parts.</figcaption>
      </figure>
    </div>
  </section>

  <section id="leakage">
    <div className="shead">
      <span className="eyebrow">05 · Leakage</span>
      <h2>Is it memorising the benchmark?</h2>
      <p className="standfirst">The obvious objection to any fine-tune, so here is the evidence rather than an assurance.</p>
    </div>

    <div className="pair full">
      <div>
        <span className="eyebrow">No benchmark part is in training</span>
        <p>The 500 benchmark parts are held out by source part, with zero shared, and every training part carrying a benchmark part's exact geometry signature was dropped. That removed 1,886 of them.</p>
      </div>
      <div>
        <span className="eyebrow">Validation loss never turned up</span>
        <p>0.212 to 0.158 across the first epoch, 0.160 to 0.149 across the second, with token accuracy reaching 94.7%. Neither pass shows the upturn you would expect from overfitting.</p>
      </div>
    </div>

    <div className="scroll full" style={{"marginTop": "var(--s7)"}}>
      <table>
        <thead><tr><th>Model</th><th>Near-duplicate (57)</th><th>No near-duplicate (443)</th></tr></thead>
        <tbody>
          <tr className="ours"><td>Cadabra, best of 8</td><td>94.7%</td><td>78.8%</td></tr>
          <tr className="ours"><td>Cadabra, 1 sample</td><td>94.7%</td><td>72.2%</td></tr>
          <tr><td>GLM-5.3 Flash (high)</td><td>84.2%</td><td>63.9%</td></tr>
          <tr><td>Kimi K3 (high)</td><td>73.7%</td><td>60.3%</td></tr>
        </tbody>
      </table>
    </div>
    <p className="note">57 of 500 benchmark parts have a training part within 1% on every bounding-box axis with the same face and part count. None within 0.1%. Those parts are easier for every model, including ones that never saw our data.</p>

    <div className="callout">
      <b>+14.9 points</b>
      <p>is our margin over the best frontier model on the 443 parts with no near-duplicate in training, slightly
      wider than the +14.5 we hold overall. Removing the parts we could plausibly have memorised makes the result
      better, not worse.</p>
    </div>
  </section>

  <section id="negative">
    <div className="shead">
      <span className="eyebrow">06 · Negative result</span>
      <h2>What didn't work</h2>
      <p className="standfirst">The grader is an exact reward, so reinforcement learning on it looked obvious. It made the model worse.</p>
    </div>

    <div className="pair full">
      <div>
        <p>We ran GRPO on top of the supervised model, 8 samples per sheet and 64 rollouts per step, for 60 steps on
        four H100s. Training reward rose from 0.89 to 1.2 and the share of correct rollouts went from 39% to 70%, so
        by its own training signal it was working.</p>
        <p>Held-out accuracy fell anyway. We graded every saved checkpoint with the same harness, including the
        supervised checkpoint as a control, which reproduced its own score to within two parts. The loss lands almost
        entirely on medium-complexity parts: 56.3% down to 48.7% after ten steps, then flat near 42%.</p>
        <p>We shipped the supervised model and published the checkpoint sweep, because a negative result with the
        evidence attached is worth more than quietly dropping the branch.</p>
      </div>
      <figure>
        <img src="/chart/rl_curve.png" alt="Held-out accuracy of each GRPO checkpoint against the supervised control" />
        <figcaption>Every GRPO checkpoint graded against the supervised control. Training reward improved while held-out accuracy fell.</figcaption>
      </figure>
    </div>
  </section>

  <section id="provenance">
    <div className="shead">
      <span className="eyebrow">07 · Provenance</span>
      <h2>The data, and what it was built on</h2>
      <p className="standfirst">Everything derives from CAD-Coder, which is Apache-2.0 and itself derived from Text2CAD and DeepCAD.</p>
    </div>

    <div className="scroll full">
      <table>
        <thead><tr><th>Set</th><th>Parts</th><th className="l">What it is</th></tr></thead>
        <tbody>
          <tr><td>Audited</td><td>82,659</td><td className="l">Every reference program run and checked against its own spec</td></tr>
          <tr className="ours"><td>Benchmark, held out</td><td>500</td><td className="l">322 simple, 119 medium, 59 complex, 43 multi-part</td></tr>
          <tr><td>Training, train_high</td><td>5,698</td><td className="l">What remains after removing benchmark sources and geometries</td></tr>
          <tr><td>Training, train_middle</td><td>24,868</td><td className="l">Two batches, weighted toward medium, complex and multi-part</td></tr>
          <tr className="ours"><td>Sheets trained on</td><td>30,260</td><td className="l">Plus 306 validation. 72% medium or complex, 35% multi-part</td></tr>
        </tbody>
      </table>
    </div>

    <div className="pair full" style={{"marginTop": "var(--s7)"}}>
      <figure>
        <img src="/chart/cost_vs_accuracy.png" alt="Cost per thousand parts against accuracy, one point per model" />
        <figcaption>Cost per 1,000 parts against accuracy. Ours includes the GPU hour amortised over measured throughput, not just tokens.</figcaption>
      </figure>
      <div>
        <span className="eyebrow">Built on Baseten</span>
        <ul className="plain" style={{"marginTop": "var(--s4)"}}>
          <li><b>Model APIs</b> for the Kimi K3 and GLM-5.3 Flash baselines, at high reasoning effort.</li>
          <li><b>Training Jobs</b> for the LoRA fine-tune. One H100 for the first epoch, four with DDP for the second.</li>
          <li><b>Deployments</b> pulling merged weights from the training job's checkpoints, served with vLLM.</li>
          <li><b>Switch</b>, which ran Claude Code on open models while we built this.</li>
        </ul>
      </div>
    </div>
  </section>

</main>
</div>

<footer>
  Built at Hack the North 2026. Data from
  <a href="https://huggingface.co/datasets/gudo7208/CAD-Coder">CAD-Coder</a> (Apache-2.0, derived from Text2CAD and
  DeepCAD). Every number on this page is reproducible from the repository.
</footer>


    </>
  );
}
