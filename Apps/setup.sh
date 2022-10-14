docker image build . -t churn_prediction:latest
docker image save --output churn_prediction.tar churn_prediction
docker container run -it --rm churn_prediction:latest