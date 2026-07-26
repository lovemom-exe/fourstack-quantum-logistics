import "../../style/main/section.css"

type SubContent = {
    title: string;
    subtitle: string;
    price: string;
    features: string[];
};


const Subscription = () => {
    const layout_style: React.CSSProperties = {
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "24px",
        width: "100%",
        maxWidth: "1200px",
        margin: "0 auto",
        padding: "40px 20px",
        boxSizing: "border-box",
    };
    return (
        <main>
            <section className="section" >
                <p className="section__eyebrow">// Select Your Subscription Plan </p>
                <p className="section__title" style={{ fontSize: "80px", fontWeight: "bold" }}>
                    SUBSCRIPTION
                </p>
                <div style={layout_style}>
                    <SubBox
                        title="STARTER"
                        subtitle="For Warehouse Managers."
                        price="$800/month"
                        features={[
                            "14-day forecast",
                            "Weather-adjusted, zero tuning",
                            "Stockout alerts before they hit",
                        ]}
                    />
                    <SubBoxBest
                        title="PROFESSIONAL"
                        subtitle="For Regional Chain Managers."
                        price="$2,500/month"
                        features={[
                            "Everythings in STARTER",
                            "Planners & procurement 30-day horizon",
                            "Basic demand forecasting",
                            "7-day forecast",
                            "Community support",
                        ]}
                    />
                    <SubBox
                        title="BUSINESS"
                        subtitle="For Third-party Logistics, National F&B and Grocery."
                        price="$60,000/Year"
                        features={[
                            "1 warehouse",
                            "Up to 100 SKUs",
                            "Basic demand forecasting",
                            "7-day forecast",
                            "Community support",
                        ]}
                    />
                    <SubBoxEnterPr
                        title="ENTERPRISE"
                        subtitle="For Multinationals & Medical Cold Chain."
                        price="$80,000/Year"
                        features={[
                            "1 warehouse",
                            "Up to 100 SKUs",
                            "Basic demand forecasting",
                            "7-day forecast",
                            "Community support",
                        ]}
                    />
                </div>

            </section>
        </main>
    )
};

export { Subscription };

// ====================================================================================
// SUBSCRIPTION BOX
// ====================================================================================

const title_style: React.CSSProperties = {
    margin: "0 0 8px",
    fontSize: "30px",
    fontWeight: 700,
};

const subtitle_style: React.CSSProperties = {
    minHeight: "60px",
    margin: "0 0 24px",
    fontSize: "14px",
    lineHeight: 1.5,
};

const price_style: React.CSSProperties = {
    margin: "0 0 24px",
    fontSize: "36px",
    fontWeight: 700,
};

const feature_list_style: React.CSSProperties = {
    margin: 0,
    paddingLeft: "20px",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
    fontSize: "15px",
    lineHeight: 1.5,

    listStyleType: "disc",
    listStylePosition: "outside",
    textAlign: "left",
    width: "100%",
};

const box_style: React.CSSProperties = {
    minHeight: "450px",
    padding: "24px",
    cursor: "pointer"
};

const SubBox = ({
    title,
    subtitle,
    price,
    features,
}: SubContent) => {

    return (
        <button className="bento-card" style={box_style}>
            <h2 style={title_style}>{title}</h2>

            <p style={subtitle_style}>{subtitle}</p>

            <p style={price_style}>{price}</p>

            <ul style={feature_list_style}>
                {features.map((feature) => (
                    <li key={feature}>{feature}</li>
                ))}
            </ul>
        </button>
    )
}

export { SubBox };

const SubBoxEnterPr = ({
    title,
    subtitle,
    price,
    features,
}: SubContent
) => {

    return (
        <button className="bento-card bento-card--dark" style={box_style}>
            <h2 style={title_style}>{title}</h2>

            <p style={subtitle_style}>{subtitle}</p>

            <p style={price_style}>{price}</p>

            <ul style={feature_list_style}>
                {features.map((feature) => (
                    <li key={feature}>{feature}</li>
                ))}
            </ul>
        </button>
    )
}

export { SubBoxEnterPr };

const SubBoxBest = ({
    title,
    subtitle,
    price,
    features,
}: SubContent) => {
    return (
        <button className="bento-card bento-card--safety" style={box_style}>
            <h2 style={title_style}>{title}</h2>

            <p style={subtitle_style}>{subtitle}</p>

            <p style={price_style}>{price}</p>

            <ul style={feature_list_style}>
                {features.map((feature) => (
                    <li key={feature}>{feature}</li>
                ))}
            </ul>
        </button>
    )
}

export { SubBoxBest };