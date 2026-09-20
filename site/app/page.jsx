import Demo from "@/components/Demo";

/* The header lives here rather than in the layout: lib/demo.js wires the two tabs by id, and the How it works page
   has its own header with links instead of tabs. */
export default function Home() {
  return (
    <>
      <header>
        <div className="inner">
          <h1>Cadabra</h1>
          <span className="sub">Engineering drawing in, CAD program out.</span>
          <nav>
            <button id="tab-compare" className="on">Compare</button>
            <button id="tab-race">Live race</button>
            <a href="/how-it-works">How it works</a>
          </nav>
        </div>
      </header>
      <Demo />
    </>
  );
}
