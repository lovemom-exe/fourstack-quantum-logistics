import { useMemo, useState } from "react";
import "../../style/data-page.css";
import { CsvUploadCard, type DatasetKey, type UploadedFile } from "./components/CsvUploadCard";
import { WarehouseForm } from "./components/WarehouseForm";
import {
    ColumnMappingTable,
    DataPreviewTable,
    DatasetSummaryCard,
    GeneratedFeatures,
    ReadinessPanel,
    UnusedColumns,
    ValidationPanel,
} from "./components/DataReview";
import { DATASETS } from "./datasets";

const initialFiles: Record<DatasetKey, UploadedFile | null> = {
    products: { name: "product_master_july.csv", size: 184_320, rows: 1248, uploadedAt: "26 Jul 2026, 09:42", status: "success" },
    suppliers: { name: "approved_suppliers.csv", size: 48_128, rows: 84, uploadedAt: "26 Jul 2026, 09:44", status: "success" },
    sales: { name: "sales_history_2024_2026.csv", size: 12_897_484, rows: 184293, uploadedAt: "26 Jul 2026, 09:48", status: "success" },
    inventory: { name: "inventory_snapshot.csv", size: 624_230, rows: 4218, uploadedAt: "26 Jul 2026, 09:51", status: "success" },
};

const Upload = () => {
    const [files, setFiles] = useState(initialFiles);
    const [warehouseComplete, setWarehouseComplete] = useState(true);
    const [previewDataset, setPreviewDataset] = useState<DatasetKey>("sales");
    const completed = Object.values(files).filter(Boolean).length + Number(warehouseComplete);
    const totalSections = 5;
    const progress = Math.round((completed / totalSections) * 100);
    const ready = warehouseComplete && Object.values(files).every(Boolean);

    const updateFile = (key: DatasetKey, file: UploadedFile | null) =>
        setFiles((current) => ({ ...current, [key]: file }));

    const attentionCount = totalSections - completed;
    const statusText = useMemo(() => attentionCount === 0 ? "All required inputs available" : `${attentionCount} required input${attentionCount === 1 ? "" : "s"} missing`, [attentionCount]);

    return (
        <main className="data-page">
            <header className="data-hero">
                <div className="data-shell data-hero__layout">
                    <div>
                        <p className="data-kicker">Demand forecasting / Data onboarding</p>
                        <h1>Data Management</h1>
                        <p className="data-hero__subtitle">Configure your warehouse and upload the operational data required for demand forecasting.</p>
                    </div>
                    <div className="setup-progress" aria-label={`Setup progress: ${progress}%`}>
                        <div className="setup-progress__top"><span>Overall setup</span><strong>{progress}%</strong></div>
                        <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
                        <p>{statusText}</p>
                        <div className="setup-progress__stats">
                            <div><strong>{completed}</strong><span>Sections complete</span></div>
                            <div><strong>{attentionCount}</strong><span>Need attention</span></div>
                        </div>
                    </div>
                </div>
            </header>

            <div className="data-shell data-content">
                <WarehouseForm complete={warehouseComplete} onCompletionChange={setWarehouseComplete} />

                <section className="data-section" aria-labelledby="upload-heading">
                    <div className="data-section__heading">
                        <div><span className="section-index">02–05</span><h2 id="upload-heading">Operational data uploads</h2><p>Upload CSV exports from your source systems. Files remain local in this prototype.</p></div>
                        <span className="status-badge status-badge--success">{Object.values(files).filter(Boolean).length}/4 uploaded</span>
                    </div>
                    <div className="upload-grid">
                        {DATASETS.map((dataset) => (
                            <CsvUploadCard key={dataset.key} dataset={dataset} file={files[dataset.key]} onFileChange={(file) => updateFile(dataset.key, file)} />
                        ))}
                    </div>
                </section>

                <section className="data-section" aria-labelledby="summary-heading">
                    <div className="data-section__heading"><div><span className="section-index">06</span><h2 id="summary-heading">Uploaded data summary</h2><p>Review file metadata and validation status before mapping fields.</p></div></div>
                    <div className="summary-grid">
                        {DATASETS.map((dataset) => <DatasetSummaryCard key={dataset.key} dataset={dataset} file={files[dataset.key]} onRemove={() => updateFile(dataset.key, null)} onPreview={() => setPreviewDataset(dataset.key)} />)}
                    </div>
                </section>

                <DataPreviewTable selected={previewDataset} onSelect={setPreviewDataset} />
                <ColumnMappingTable />
                <UnusedColumns />
                <ValidationPanel />
                <GeneratedFeatures />
                <ReadinessPanel ready={ready} warehouseComplete={warehouseComplete} uploadedCount={Object.values(files).filter(Boolean).length} />
            </div>
        </main>
    );
};

export { Upload };
