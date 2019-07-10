FROM oracle/openjdk:8
WORKDIR /opt
RUN mkdir data
RUN mkdir target
ADD bootstrap-executable-1.0-SNAPSHOT.jar .
ADD spam_out.csv .
ADD create_model.sh .
ADD test_model.sh .
RUN chmod u+x *.sh