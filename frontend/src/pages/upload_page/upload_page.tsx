import "../../style/main/section.css";
import "../../style/elements/card.css";

const Upload = () => {
    return (
        <main>
            <section className="section">
                <div className="section__inner">
                    <div className="section__header section__left">
                        <p className="section__eyebrow">// Forecast Your Customer Demand</p>
                        <p className="section__title">YOUR WAREHOUSE DATA</p>
                        <div className="section__header">
                            Insert Your New Warehouse Data To Predict Your Customer's Demand
                        </div>
                    </div>
                    <div className="section__left bento__grid">
                        <div className="bento-card"></div>
                    </div>
                </div>
            </section>
            <section className="section section--dark">

            </section>
        </main>
    )
};

export { Upload };