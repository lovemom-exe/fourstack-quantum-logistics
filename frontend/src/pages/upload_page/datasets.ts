import type { DatasetKey } from "./components/CsvUploadCard";

export type DatasetDefinition = {
    key: DatasetKey;
    name: string;
    description: string;
    columns: string[];
};

export const DATASETS: DatasetDefinition[] = [
    {
        key: "products", name: "Products", description: "Product master data, pricing, shelf life, and storage requirements.",
        columns: ["product_id", "product_name", "category", "shelf_life_days", "storage_temp", "spoilage_sensitivity", "base_price", "cost_price", "supplier_id"],
    },
    {
        key: "suppliers", name: "Suppliers", description: "Supplier performance, lead times, and order constraints.",
        columns: ["supplier_id", "supplier_name", "supplier_score", "lead_time_days", "minimum_order_quantity", "supplier_country"],
    },
    {
        key: "sales", name: "Sales", description: "Historical transactions used to learn customer demand.",
        columns: ["transaction_date", "product_id", "store_id", "units_sold", "selling_price", "discount_pct", "is_promoted"],
    },
    {
        key: "inventory", name: "Inventory", description: "Current stock, incoming quantities, batches, and expiration dates.",
        columns: ["product_id", "warehouse_id", "current_inventory", "reserved_inventory", "incoming_inventory", "expiration_date", "batch_id"],
    },
];

export const formatFileSize = (bytes: number) =>
    bytes > 1_000_000 ? `${(bytes / 1_000_000).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
