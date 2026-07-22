import "../../style/elements/card.css";


interface StatCardProps {
    label?: string;
    data?: string;
    unit?: string;
}

const StatCard1 = ({ label, data, unit }: StatCardProps) => {
    return (
        <div className="bento-card bento-card--mid">
            <div className="stat-card__label">{label ? label : "non"}</div>
            <div className="stat-card__value">{data ? data : 0} <span className="stat-card__unit">{unit ? unit : "..."}</span>
            </div>
        </div>
    )
};

export { StatCard1 };

const StatCard2 = ({ label, data, unit }: StatCardProps) => {
    return (
        <div className="bento-card bento-card--mid bento-card--dark">
            <div className="stat-card__label">{label ? label : "non"}</div>
            <div className="stat-card__value">{data ? data : 0} <span className="stat-card__unit">{unit ? unit : "..."}</span>
            </div>
        </div>
    )
};

export { StatCard2 };

const StatCard3 = ({ label, data, unit }: StatCardProps) => {
    return (
        <div className="bento-card bento-card--mid bento-card--safety">
            <div className="stat-card__label">{label ? label : "non"}</div>
            <div className="stat-card__value">{data ? data : 0} <span className="stat-card__unit">{unit ? unit : "..."}</span>
            </div>
        </div>
    )
};

export { StatCard3 };