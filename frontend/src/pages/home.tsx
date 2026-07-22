// import React from "react";
import { Hero } from "../pages/hero";
import { PreviewLabel } from "../pages/preview_label";

const Home = () => {
  return (
    <>
      <Hero />
      <PreviewLabel />

      <section className="section">
        <div className="container">
          <div className="section__header">
            <p className="section__eyebrow">// Demand Forecasing</p>
            <h1 className="section__title">OverView About <br />Our Project</h1>
          </div>

          <div className="bento__grid">
            <div className="bento-card bento-card--wide reveal">
              <p className="bento-card__number">no.01</p>
              <h1 className="bento-card__title"></h1>
              <h6 className="bento-card__text">

              </h6>
            </div>

            <div className="bento-card bento-card--narrow bento-card--safety">
              <p className="bento-card__number">no.02</p>
              <h1 className="bento-card__title"></h1>
              <h6>
                <ul className="bento-card__text">
                  <li></li>
                  <li></li>
                  <li></li>
                  <li></li>
                </ul>
              </h6>
            </div>

            <div className="bento-card bento-card--mid bento-card--dark">
              <p className="bento-card__number">no.03</p>
              <h1 className="bento-card__title"></h1>
              <h6>
                <p className="bento-card__text">
                </p>
              </h6>
            </div>
            <div className="bento-card bento-card--mid">
              <p className="bento-card__number">no.04</p>
              <h1 className="bento-card__title"></h1>
              <h6>
                <p className="bento-card__text">
                </p>
              </h6>
            </div>
          </div>
        </div>
      </section>
    </>
  )
};

export { Home };
