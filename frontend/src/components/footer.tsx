import "../style/main/footer.css";
import type {Brand} from "./header.tsx";

const Footer = ({myBrand}: {myBrand: Brand}) =>
{
    return (
        <footer className="footer">
            <div className="footer__inner">
                <div className="footer__top">
                    <div className="footer__brand">
                        <h2 className="font-display footer__brand-name">
                            {myBrand.brand_name ? myBrand.brand_name : "Your Brand Name"}
                            <span className="accent">.</span>
                        </h2>
                        <p className="footer__brand-desc">
                            {myBrand.brand_desc ? myBrand.brand_desc : "Add Description Here"}
                        </p>
                    </div>
                    <div className="footer__col">
                        <div className="footer__col-title">Services</div>
                        <a href="#">SmartFarm AI Weather ChatBot</a>
                        <a href="#">Air Quality Prediction</a>
                        <a href="#">Climate Change</a>
                    </div>
                    <div className="footer__col">
                        <div className="footer__col-title">Company</div>
                        <a href={myBrand.linkedin? myBrand.linkedin : "#"} target="_blank" rel="noopener noreferrer">About Me</a>
                        <a href={myBrand.github? myBrand.github : ""} target="_blank" rel="noopener noreferrer">Projects</a>
                    </div>
                    <div className="footer__col">
                        <div className="footer__col-title">Get in Touch</div>
                        <a href="#">{myBrand.email? myBrand.email : "mymail@gmail.com"}</a>
                        <a href={myBrand.facebook? myBrand.facebook : ""} target="_blank" rel="noopener noreferrer">{myBrand.facebook && "facebook"}</a>
                        <a href={myBrand.twitter? myBrand.twitter : ""} target="_blank" rel="noopener noreferrer">{myBrand.twitter && "Twitter"}</a>
                        <a href={myBrand.linkedin? myBrand.linkedin : ""} target="_blank" rel="noopener noreferrer">{myBrand.linkedin && "LinkedIn"}</a>
                        <a href={myBrand.github? myBrand.github : ""} target="_blank" rel="noopener noreferrer">{myBrand.github && "GitHub"}</a>
                    </div>
                </div>
                <div className="footer__bottom">
                    <span className="footer__copy">&copy; {myBrand.footer_title ? myBrand.footer_title : "Your Brand Name"} &nbsp;|&nbsp; Created by <a href={myBrand.author_name ? myBrand.author_name : "https://www.linkedin.com/in/ho%C3%A0ng-qu%C3%A2n-652a9b2b0/"} target="_blank" rel="nofollow noopener">{myBrand.author_name ? myBrand.author_name : "Quan Hoang"}</a>
                    </span>
                    <button className="footer__back-top" id="backToTop" aria-label="Back to top">↑ Back to Top</button>
                </div>
            </div>
        </footer>
    )
};

export {Footer};