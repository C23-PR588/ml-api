# AutoForex ml-api

Endpoint: https://ml-dot-nomadic-grid-382206.et.r.appspot.com

This Endpoint will be called in Back-end.

## Predict 

### Preduct Currency Percentage in 7 day URL /ml/predict{currencyName_in_capitalize}
* Method get

Example output JSON with AUD currency /ml/predictAUD
```json
{
    "error": false,
    "message": "success",
    "data": {
        "value": 1.92
    }
}
```
Response Json like above (if Preduct Currency Percentage is successful)

```json
{
    "error": true,
    "message": Error message
}
```
Response Json like above (if Preduct Currency Percentage is not success)
