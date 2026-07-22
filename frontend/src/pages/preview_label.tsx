import "../style/elements/ticker.css";

const PreviewLabel = () => {
    return (
        <div className="ticker ticker--safety">
        <div className="ticker__track">
          <div className="ticker__item">Temperature</div>
          <div className="ticker__item">Humidity</div>
          <div className="ticker__item">Pressure</div>
          <div className="ticker__item">Wind Speed</div>
          <div className="ticker__item">PM2.5</div>
          <div className="ticker__item">PM10</div>
          <div className="ticker__item">Ozone</div>
        </div>
      </div>
    )
};

export {PreviewLabel};