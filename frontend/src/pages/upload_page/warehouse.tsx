const Warehouse = () => {
    const form_style: React.CSSProperties = {
        display: "grid",
        gridTemplateColumns: '120px 1fr',
        gap: '16px',
        maxWidth: '1000px',
        padding: '20px',
    };
    const label_style: React.CSSProperties = {
        display: 'flex',
        alignItems: 'center',
        fontWeight: 'bold',
        fontSize: '20px',
        maxWidth: "100px"
    };

    const input_style: React.CSSProperties = {
        padding: '10px',
        height: '60px',
        width: '100%',
        fontSize: "20px",
    };
    return (
        <section className="section">
            <div className="section__inner">
                <div className="section__header section__left">
                    <p className="section__eyebrow">// Forecast Your Customer Demand</p>
                    <p className="section__title">YOUR WAREHOUSE DATA</p>
                    <div className="section__header">
                        Insert Your New Warehouse Data
                    </div>
                </div>
                <div className="section__left ">
                    <form style={form_style}>
                        <label style={label_style}>Warehouse Name:</label>
                        <input type="text" style={input_style} />

                        <label style={label_style}>Location:</label>
                        <select className="bento-card" style={input_style} name="Location" id="">
                            <option value="">Region 1</option>
                            <option value="">Region 2</option>
                            <option value="">Region 3</option>
                        </select>

                        <label style={label_style}>Warehouse Type:</label>
                        <select className="bento-card" style={input_style} name="Location" id="">
                            <option value="">Ambient</option>
                            <option value="">Cold</option>
                            <option value="">Frozen</option>
                            <option value="">Mixed</option>
                        </select>

                        <label style={label_style}>Storage Capacity:</label>
                        <input type="number" style={input_style} />
                    </form>
                </div>
            </div>
        </section>
    )
};

export { Warehouse };