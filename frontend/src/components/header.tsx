// import { Route } from "lucide-react";
import { Link } from "react-router-dom";
import "../style/main/header.css";

interface Brand {
    brand_name?: string;
    brand_logo?: string;
    brand_desc?: string;
    email?: string;
    facebook?: string;
    twitter?: string;
    linkedin?: string;
    github?: string;
    footer_title?: string;
    author_name?: string;
}
export type { Brand };

const Header = ({ myBrand }: { myBrand: Brand }) => {
    return (
        <header className="header">
            <div className="header__inner">
                <div className="header__left">
                    <a href="/" className="header__logo" aria-label="AXIS INDUSTRIAL Home">
                        <span className="header__logo-mark"></span>
                        {myBrand.brand_name ? myBrand.brand_name : "Your Brand Name"}
                    </a>
                </div>

                <nav className="header__nav" id="headerNav">
                    <Link to="/">Home Map</Link>
                    <Link to="/data">Data</Link>
                    <Link to="/forecast">Forecast</Link>
                </nav>

                <div className="header__right">
                    <div className="header__status">
                        <span className="header__status-dot"></span>
                        <span>Online</span>
                    </div>
                    <div className="header__clock" id="utcClock">--:--:-- UTC</div>
                </div>
            </div>
        </header>
    );
};

export { Header };
