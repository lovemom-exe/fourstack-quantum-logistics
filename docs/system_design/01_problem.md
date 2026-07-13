# Problem Statement

## Background

To better understand real operational challenges, we interviewed several small businesses in Hanoi that distribute fresh food and perishable products, including **BOVEN+** and other local companies. Although their business models differ, they consistently reported similar operational difficulties.

Most companies still rely on spreadsheets, manual calculations, and personal experience when making inventory and warehouse planning decisions. As business volume grows, these approaches become increasingly inefficient and difficult to scale.

These observations led us to identify two common problems shared across many logistics and supply chain operations, rather than problems unique to a single company.

---

## Current Workflow

Warehouse managers typically perform the following tasks:

1. Collect historical sales and inventory data.
2. Estimate future customer demand.
3. Plan inventory replenishment.
4. Allocate products within warehouse capacity.
5. Coordinate procurement and daily warehouse operations.
6. Monitor inventory levels and product movement.

Most of these decisions are performed manually with limited analytical support.

---

# Core Problem

## Sub-problem 1: Demand Forecasting

Accurately predicting future demand is challenging because customer demand continuously changes due to factors such as:

* Seasonal trends
* Promotional campaigns
* Market fluctuations
* Customer purchasing behavior

Inaccurate forecasts often result in either overstocking or stock shortages, both of which negatively impact operational efficiency and business performance.

---

## Sub-problem 2: Warehouse Storage & Goods Allocation Optimization

Warehouse managers must allocate products while considering multiple operational constraints, including:

* Limited warehouse storage capacity
* Product shelf life
* Inventory balance across different product categories
* Space utilization
* Procurement timing
* Operational costs

Finding an optimal allocation strategy is difficult because improving one objective often negatively affects another. As a result, warehouse planning becomes a complex optimization problem that is difficult to solve manually.

---

## Proposed Solution

Develop an AI-assisted logistics optimization platform that addresses both identified sub-problems through quantum-enhanced computing techniques:

* **Demand Forecasting:** Develop a Quantum Machine Learning (QML) forecasting model (e.g., Variational Quantum Classifier (VQC), Variational Quantum Regressor (VQR), or Quantum Neural Network (QNN)) to predict future product demand from historical sales and operational data.
* **Warehouse Storage & Goods Allocation Optimization:** Formulate warehouse planning as a combinatorial optimization problem and solve it using quantum optimization algorithms such as the **Quantum Approximate Optimization Algorithm (QAOA)** to generate efficient storage and goods allocation strategies under operational constraints.
* **Decision Support:** Provide warehouse managers with data-driven recommendations for inventory planning and warehouse operations through an intuitive web-based interface.
* **Scalable Architecture:** Design the system with a modular architecture that separates the frontend, backend, forecasting, and optimization components, enabling future improvements or replacement of individual models without affecting the overall system.

---

## Target Users

### Primary Users

* Warehouse Manager
* Logistics Operations Manager

### Secondary Users

* Business Administrator
* Operations Supervisor

The initial target market focuses on small and medium-sized logistics businesses in Hanoi, with the architecture designed to scale to organizations in other regions and industries.
