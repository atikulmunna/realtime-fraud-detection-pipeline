FROM flink:1.19.1-scala_2.12-java17

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*
COPY infra/flink/requirements.txt /opt/fraud/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /opt/fraud/requirements.txt
ADD https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.2.0-1.19/flink-sql-connector-kafka-3.2.0-1.19.jar /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar
RUN chmod 0444 /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar
COPY src /opt/fraud/src
COPY schemas /opt/fraud/schemas
ENV PYTHONPATH=/opt/fraud
USER flink
