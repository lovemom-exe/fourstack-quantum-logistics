import "../../style/main/section.css";
import "../../style/elements/card.css";
import { Warehouse } from "./warehouse"
import { HeroUpload } from "./hero_upload"

const Upload = () => {
    return (
        <main>
            <HeroUpload />

            <Warehouse />
            <section className="section section--dark">

            </section>
        </main>
    )
};

export { Upload };