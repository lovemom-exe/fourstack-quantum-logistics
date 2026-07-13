
# 1. Problem

## Cold Chain Import Planning Optimization

In Southeast Asia’s tropical climate, demand for perishables is rapidly growing. Importers must place orders **30–90 days in advance**, creating a long lead time during which demand is highly uncertain. A wrong import decision can cause either **overstock** (products expire and spoil, warehouse congestion, high energy costs) or **stockouts** (lost sales, emergency re-shipment costs, etc...). 

Import planning in this context is a **massively complex combinatorial optimization**. Every SKU, supplier, shipment, and storage assignment is a decision variable, all subject to constraints like shelf-life, cold-warehouse capacity, budgets, and service-level targets. As the number of products, suppliers, and warehouses increases, the number of possible import plans grows **exponentially**, overwhelming classical algorithms. In practice, planners often rely on heuristics or manual rules, which leads to wasted inventory or missed demand (90% of SEA’s food waste occurs in transit).

Our approach treats **import planning as a single joint optimization**: simultaneously balancing forecast uncertainty, lead times, inventory holding costs, shelf-life, and warehouse capacity. This creates thousands of interdependent binary decisions. Quantum optimization methods (QAOA, quantum annealing) are *specifically suited* for such high-dimensional, NP-hard problems. Quantum annealers excel at exploring enormous solution spaces of many variables and constraints. In fact, leading research shows quantum approaches applied to routing, inventory slotting, and demand forecast can outperform classical-only methods on large instances. 

*Key pain points:* Demand volatility, long lead-times, perishable shelf-life, and capacity constraints create a **combinatorial decision space** too large for traditional planning systems. (For example, assigning 10,000 SKUs across 50 warehouses and 20 suppliers with varying lot sizes and expirations produces an astronomical number of combinations.) Hybrid quantum–classical algorithms can tackle this scale by rapidly generating good candidate solutions under time pressure. 

*This unified import planning problem (forecast + ordering + storage) directly addresses the major cold-chain logistics challenges of spoilage and stockouts, integrating the conflicting objectives that existing systems handle separately.*

---

# 2. Market Size

**Business Model:** Annual SaaS subscription (USD 24,000 per company).

- **TAM (Total Addressable Market):**  ~2,000 cold-chain importers/distributors across ASEAN × \$24K = **\$48M/yr**.  
  (By comparison, industry reports value the ASEAN cold-chain logistics market at ~~\$10B in 2025~~, growing ~8%/yr, so our TAM is a small but realistic niche within this multi‑billion dollar sector.)

- **SAM (Serviceable Available Market):** ~500 target firms in Vietnam & Thailand × \$24K = **\$12M/yr**. These include mid-size food and pharma importers and 3PLs with cold storage.

- **SOM (Serviceable Obtainable Market):** Initial focus on top 20 mid-size cold-chain companies in VN/TH × \$24K = **\$480K/yr**. 

*Market context:* ASEAN cold-chain logistics is a rapidly growing industry (USD 8.6B in 2023, ~13.7B by 2029). Demand for perishables (meat, produce, vaccines) is surging with urbanization, so large players will invest in software to reduce waste and stockouts. Our TAM of \$48M is conservative relative to the overall sector size.

---

# 3. Target Client

**Primary Decision-Makers:**  
- **Import Planning Manager:** Plans international orders. Responsible for multi-month purchase decisions, forecasting import demand, and balancing incoming shipments with cold storage. Must prevent excess spoilage and avoid stockouts under uncertainty.  
- **Supply Chain/Logistics Director:** Oversees end-to-end supply chain. Focuses on inventory optimization, warehouse utilization, cold-chain costs, and on-time delivery. Seeks strategic tools for demand-driven procurement and cost reduction.

**Target Companies:** Medium-sized firms in the cold-chain space, e.g.:  
- Cold-chain 3PL providers  
- Frozen food / seafood importers  
- Fresh produce distributors  
- Pharmaceutical and vaccine distributors  

Geography: Primarily **Vietnam** and **Thailand** (ASEAN expansion later). These roles and firms directly experience the outlined pain points and have budget capacity for specialized supply-chain software.  

By addressing a core decision (import quantities under uncertainty and constraints), our solution aligns with the needs of these stakeholders and the scale of their operations.
