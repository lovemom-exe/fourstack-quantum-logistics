import { createRoot } from "react-dom/client";
import { StrictMode } from "react";
// ============================================================
// IMPORT
// ============================================================
// src/main.tsx
import { BrowserRouter, Routes, Route } from "react-router-dom";

// Stylesheets
import "./style/main/section.css";
import "./style/elements/card.css";
import "./style/index.css";

// Components & Layout Layers
import { Header } from "./components/header";
import { Footer } from "./components/footer";
import type { Brand } from "./components/header";

// Pages
import { Home } from "./pages/home";
import { DemandForecast } from "./pages/demand_forecast/demand_forecast";
import { Upload } from "./pages/upload_page/upload_page";

// ============================================================
// OUR BRAND
// ============================================================
const MyBrand: Brand = {
  brand_name: "four-stack devs",
  // brand_logo = "",
  brand_desc: "Personal Project",

  email: "quanhoang1026@gmail.com",
  // linkedin: "https://www.linkedin.com/in/ho%C3%A0ng-qu%C3%A2n-652a9b2b0/",
  // github: "https://github.com/quanhoang55",

  footer_title: "four-stack devs",
  author_name: "Quan & Hoang",
}
// ============================================================
// MAIN
// ============================================================
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>

      <Header myBrand={MyBrand} />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/data" element={<Upload />} />
        <Route path="/forecast" element={<DemandForecast />} />
      </Routes>
      <Footer myBrand={MyBrand} />

    </BrowserRouter>
  </StrictMode>,
);
