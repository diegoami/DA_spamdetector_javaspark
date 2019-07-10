FROM oracle/openjdk:8
WORKDIR /opt
RUN mkdir data
RUN mkdir target
ADD target/bootstrap-executable-1.0-SNAPSHOT.jar target/
ADD spam_out.csv .
ADD create_model.sh .
ADD test_model.sh .
RUN chmod u+x *.sh