// ============================================================
// IMPORT
// ============================================================
import "../style/main/hero.css";


// ============================================================
// HERO
// ============================================================
const Hero = () => {
    return (
        <section className="hero">
            <div className="hero__grid hero__grid--dense" aria-hidden="true"></div>
            <div className="hero__label reveal">Est. 2026 — Demand Forecasting</div>
            <h1 className="hero__title">
                <span className="line">PREDICT</span>
                <span className="line line--indent">DEMAND<span className="accent-dot">.</span></span>
            </h1>
            {/* <div>
                <WorldMap />
            </div> */}
            <div className="hero__bottom">
                <p className="hero__desc reveal">This Is Our Hackathon Project In Quantum Machine learning Algorithms</p>
                <div className="hero__stats reveal">
                    {/* <div><div className="hero__stat-num">340<span className="accent">+</span></div><div className="hero__stat-label">Projects Delivered</div></div>
                    <div><div className="hero__stat-num">99.7<span className="accent">%</span></div><div className="hero__stat-label">Uptime Record</div></div>
                    <div><div className="hero__stat-num">16</div><div className="hero__stat-label">Countries Served</div></div> */}
                </div>
            </div>
            <div className="hero__scroll-cue" aria-hidden="true">Scroll</div>
        </section>
    )
};
export { Hero };