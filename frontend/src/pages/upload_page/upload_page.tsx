import "../../style/main/section.css"
import "../../style/elements/card.css"
import "../../style/elements/button.css"
import '../../style/index.css'
import { Warehouse } from "./warehouse"
import { HeroUpload } from "./hero_upload"
import { Product } from './product'

const Upload = () => {
    return (
        <main>
            <HeroUpload />

            <Warehouse />
            <Product />
            <section className="section section--dark">

            </section>
        </main>
    )
};

export { Upload };