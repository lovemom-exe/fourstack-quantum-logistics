# Use Case Diagram

``` mermaid
graph TD

    subgraph ContainerBox [Logistics Optimization]
        uc1([Upload sale data])

        uc2([Request Demand Forecast])

        uc3([Run Warehouse Optimization])
        
        uc4([View Optimization Result])

        uc3 -.include.-> uc2
        uc4 -.include.-> uc3

        uc5([View dashsboard])

        uc2 -.include.-> uc1


    end

    u1[Warehouse manager]
    u1 --- uc1
    u1 --- uc2
    u1 --- uc3
    u1 --- uc4

    u2[Admin]
    u2 --- uc5
    u2 --- uc1
```
