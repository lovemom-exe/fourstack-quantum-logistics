const Product = () => {
    return (
        <section className="section section--safety ">
            <p className="section__eyebrow" style={{ fontSize: "15px", color: "var(--ink)" }}>// Forecast Your Customer Demand</p>
            <p className="section--title" style={{ fontSize: "70px", color: "var(--ink)", fontWeight: "bold" }}> Upload Your Data</p>
            <div className="bento__grid">
                <UploadBox title="PRODUCTS" />
                <UploadBox title="SUPPLY" />
                <UploadBox title="SALE HISTORY" />
                <UploadBox title="INVENTORY" />
            </div>
        </section>
    )
}

export { Product }


type uploadTitle = {
    title: string;
}

const handleUpload = (
    event: React.ChangeEvent<HTMLInputElement>,
) => {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
        alert("Please upload a CSV file.");
        event.target.value = "";
        return;
    }

    console.log("Uploaded CSV:", file);
};

export const UploadBox = ({
    title
}: uploadTitle) => {
    return (
        <label className="bento-card__nonshadow bento-card--mid" style={{ cursor: "pointer" }}>
            <span style={{ fontSize: "32px" }}>↑</span>

            <strong>Upload your CSV file</strong>

            <span style={{ fontSize: "50px", fontWeight: "black" }}>
                {title}
            </span>

            <input
                type="file"
                accept=".csv,text/csv"
                onChange={handleUpload}
                style={{ display: "none" }}
            />
        </label>
    )
}