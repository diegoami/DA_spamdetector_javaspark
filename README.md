SPAM DETECTOR using JAVA SPARK
=========================

A model created in Java Spark to classifying mails to spam or not

## SET UP

Make sure you have Java 8 or higher installed and Maven

## COMPILE

Execute `mvn clean compile package`

## RUN locally

To create the model and export it to `data/sparkmodel`, execute the command

```
java -cp target/bootstrap-executable-1.0-SNAPSHOT.jar com.amicabile.spamclassifier.CreateModel
# /bin/sh create_model.sh
```

To tes

```
java -cp target/bootstrap-executable-1.0-SNAPSHOT.jar com.amicabile.spamclassifier.TestModel
# /bin/sh test_model.sh
```


## EXECUTE IN DOCKER

YOUR_DATA_DIRECTORY is where you put the model files

```
docker build -t spam_detector_javaspark . 
docker run  -v $(pwd)/data:/data spam_detector_pmml /bin/sh create
# docker run  -v <YOUR_DATA_DIRECTORY>:/data spam_detector_pmml
```
