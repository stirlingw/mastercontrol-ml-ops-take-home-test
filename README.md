# mastercontrol-ml-ops-take-home-test

# Take home test instructions


### 1. Build the docker image

```
make build
```

### 2. Run the container

```
make run 
```

### 3. Run Pytest with coverage
```
make pytest
```

### 4. Go to Swagger documentation to run API calls
[http://localhost:8000/docs](http://localhost:8000/docs)


### 5. Try out the post /predict_text method
```
curl -X 'POST' \
  'http://localhost:8000/predict_text/predict_text' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "file_name": "tht-sample-text-data.csv"
}'
```

### 6. Try out the post /predict_tabular method
```
curl -X 'POST' \
  'http://localhost:8000/predict_tabular/predict_tabular' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "file_name": "tht-test-tabular-data.csv"
}'
```

# Test Notes
If working with csv files I would include a s3 path per client/model/date for getting data.
A better idea might be to figure out a way to use S3/Athena to get data or store data in a database. 
I would also consider going with kafka queue over API if working with csv files


## Further Work List
1. Finish adding pytests or tox for unit testing
2. Clean up functions in predict_text_handler.py
3. Add locust tests for stress testing API
4. Get test API onto AWS or GCP

## Test Feedback

1. Include README.md in the initial zip file
2. Reduce scope of the take home test
3. Include test data for text multiclass problem
4. Include suggested endpoint names
5. Include expected input and output of api endpoints
6. Have multiple MLE/Scientist peer review the test
7. Suggest Google Colab instead of expecting an MLOPS candidate to put API in GCP or AWS 