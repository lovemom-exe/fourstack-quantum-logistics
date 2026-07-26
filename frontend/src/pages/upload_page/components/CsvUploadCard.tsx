import { useRef, useState, type DragEvent } from "react";
import { formatFileSize } from "../datasets";
import type { DatasetDefinition } from "../datasets";

export type DatasetKey = "products" | "suppliers" | "sales" | "inventory";
export type UploadedFile = { name: string; size: number; rows: number; uploadedAt: string; status: "success" | "error" };

type Props = { dataset: DatasetDefinition; file: UploadedFile | null; onFileChange: (file: UploadedFile | null) => void };
export const CsvUploadCard = ({ dataset, file, onFileChange }: Props) => {
    const inputRef = useRef<HTMLInputElement>(null);
    const [dragging, setDragging] = useState(false);
    const [error, setError] = useState("");
    const [progress, setProgress] = useState(file ? 100 : 0);

    const acceptFile = (selected?: File) => {
        if (!selected) return;
        if (!selected.name.toLowerCase().endsWith(".csv")) {
            setError("Only .csv files are accepted.");
            setProgress(0);
            return;
        }
        setError("");
        setProgress(55);
        window.setTimeout(() => {
            setProgress(100);
            onFileChange({ name: selected.name, size: selected.size, rows: Math.max(1, Math.round(selected.size / 74)), uploadedAt: new Date().toLocaleString(), status: "success" });
        }, 450);
    };
    const drop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]); };
    const template = () => {
        const blob = new Blob([`${dataset.columns.join(",")}\n`], { type: "text/csv" });
        const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `${dataset.key}_template.csv`; link.click(); URL.revokeObjectURL(link.href);
    };

    return (
        <article className="upload-card">
            <div className="upload-card__title"><div className="dataset-icon">{dataset.name.slice(0, 2).toUpperCase()}</div><div><h3>{dataset.name}</h3><p>{dataset.description}</p></div></div>
            {dataset.key === "sales" && <div className="target-note"><strong>Prediction target</strong><code>units_sold</code> from historical records</div>}
            <div className={`drop-zone ${dragging ? "drop-zone--active" : ""} ${error ? "drop-zone--error" : ""}`} onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={drop}>
                <input ref={inputRef} type="file" accept=".csv,text/csv" onChange={(e) => acceptFile(e.target.files?.[0])} />
                <span className="drop-zone__icon" aria-hidden="true">↑</span>
                {file ? <><strong>{file.name}</strong><span>{formatFileSize(file.size)} · CSV file</span></> : <><strong>Drop CSV here or <button type="button" className="text-button" onClick={() => inputRef.current?.click()}>browse</button></strong><span>Maximum file size 50 MB</span></>}
                {progress > 0 && progress < 100 && <div className="upload-progress"><span style={{ width: `${progress}%` }} /></div>}
                {error && <span className="field-error" role="alert">{error}</span>}
            </div>
            <details className="expected-columns"><summary>Expected columns ({dataset.columns.length})</summary><div>{dataset.columns.map((column) => <code key={column}>{column}</code>)}</div></details>
            <div className="card-actions">
                <button type="button" className="data-button data-button--quiet" onClick={template}>Download template</button>
                {file && <><button type="button" className="data-button" onClick={() => inputRef.current?.click()}>Replace</button><button type="button" className="icon-button icon-button--danger" onClick={() => { onFileChange(null); setProgress(0); }} aria-label={`Remove ${dataset.name} file`}>×</button></>}
            </div>
        </article>
    );
};
