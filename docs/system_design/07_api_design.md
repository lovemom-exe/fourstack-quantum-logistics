# API Specification

# API Specification

| Method | Endpoint | Purpose | Request Body | Success Response | Error Response |
|--------|----------|---------|--------------|------------------|----------------|
| POST | /sales | upload sale data | JSON | 201 Created | 400 500 |
| POST | /forecast | resquest to forecast the demand | None | forecast result | 400 500 |
| GET | /sales | get sale data from DATABASE | None | sale data| 400 500 |
| GET | /forecast | Retrieve all forecast data | None | forecast result | 400 500 |
