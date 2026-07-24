const HeroUpload = () => {
    return (
        <section className="hero">
            <div className="hero__grid hero__grid--dense" aria-hidden="true"></div>
            <div className="hero__label reveal">Your Warehouse Data</div>
            <h1 className="hero__title">
                <span className="line">Warehouse</span>
                <span className="line line--indent">Data<span className="accent-dot">.</span></span>
            </h1>
            <div className="hero__bottom">
                <p className="hero__desc ">Your warehouse data will be used to predict your customer's demand, You can add more than one warehouse base on your subscription. And your data will be secure!</p>
                {/* <div className="hero__desc reveal">
                    <div><div className="hero__stat-num bento-card">warehouse</div></div>
                    <div><div className="hero__stat-num bento-card">product</div></div>
                    <div><div className="hero__stat-num bento-card">supplier</div></div>
                    <div><div className="hero__stat-num bento-card">sale history</div></div>
                    <div><div className="hero__stat-num bento-card">inventory</div></div>
                </div> */}
            </div>
        </section>
    )
};

export { HeroUpload };