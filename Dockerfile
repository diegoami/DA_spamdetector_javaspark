FROM oracle/openjdk:8
WORKDIR /opt
RUN mkdir data
RUN mkdir target
ADD spam_out.csv .
ADD create_model.sh .
ADD test_model.sh .
ADD BOT* ./
RUN cat BOT* > bootstrap-executable-1.0-SNAPSHOT.jar
RUN rm -f BOT*
RUN chmod u+x *.sh