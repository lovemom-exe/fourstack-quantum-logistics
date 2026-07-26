import { useState, type FormEvent } from "react";

type Props = { complete: boolean; onCompletionChange: (complete: boolean) => void };
const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

export const WarehouseForm = ({ complete, onCompletionChange }: Props) => {
    const [editing, setEditing] = useState(!complete);
    const [operatingDays, setOperatingDays] = useState(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]);
    const submit = (event: FormEvent) => { event.preventDefault(); setEditing(false); onCompletionChange(true); };
    return (
        <section className="data-section" aria-labelledby="warehouse-heading">
            <div className="data-section__heading">
                <div><span className="section-index">01</span><h2 id="warehouse-heading">Warehouse information</h2><p>Set the operating context used for inventory and demand calculations.</p></div>
                <div className="heading-status"><span className={`status-badge ${complete ? "status-badge--success" : "status-badge--warning"}`}>{complete ? "Complete" : "Needs attention"}</span><small>Last updated 26 Jul 2026, 09:30</small></div>
            </div>
            <form className="warehouse-form" onSubmit={submit}>
                <label><span>Warehouse name</span><input required disabled={!editing} defaultValue="Central Fresh Distribution" /></label>
                <label><span>Country</span><select disabled={!editing} defaultValue="Vietnam"><option>Vietnam</option><option>Singapore</option><option>Thailand</option></select></label>
                <label><span>Region</span><input required disabled={!editing} defaultValue="Southeast" /></label>
                <label><span>City</span><input required disabled={!editing} defaultValue="Ho Chi Minh City" /></label>
                <label><span>Warehouse type</span><select disabled={!editing} defaultValue="Mixed"><option>Ambient</option><option>Chilled</option><option>Frozen</option><option>Mixed</option></select></label>
                <label><span>Storage capacity</span><input required disabled={!editing} type="number" min="0" defaultValue="12500" /></label>
                <label><span>Capacity unit</span><select disabled={!editing} defaultValue="Pallets"><option>Pallets</option><option>Cubic meters</option><option>Units</option></select></label>
                <label><span>Default currency</span><select disabled={!editing} defaultValue="VND"><option>VND</option><option>USD</option><option>SGD</option></select></label>
                <label className="form-span-2"><span>Time zone</span><select disabled={!editing} defaultValue="Asia/Ho_Chi_Minh"><option>Asia/Ho_Chi_Minh</option><option>Asia/Singapore</option><option>UTC</option></select></label>
                <fieldset className="form-span-2" disabled={!editing}><legend>Operating days</legend><div className="day-picker">{days.map(day => <label key={day}><input type="checkbox" checked={operatingDays.includes(day)} onChange={() => setOperatingDays(current => current.includes(day) ? current.filter(item => item !== day) : [...current, day])} /><span>{day}</span></label>)}</div></fieldset>
                <div className="form-actions form-span-2">
                    {!editing ? <button type="button" className="data-button" onClick={() => setEditing(true)}>Edit configuration</button> : <><button type="button" className="data-button data-button--quiet" onClick={() => setEditing(false)}>Cancel</button><button type="submit" className="data-button data-button--primary">Save warehouse</button></>}
                </div>
            </form>
        </section>
    );
};
