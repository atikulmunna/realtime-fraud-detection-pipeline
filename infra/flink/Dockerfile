FROM flink:1.19.1-scala_2.12-java17

USER root
# Flink's PythonDriver shells out to `python`, not `python3`, and Ubuntu 22.04
# ships no such alias. Without the symlink `flink run --python` fails with
# 'Cannot run program "python": error=2, No such file or directory' and the job
# never reaches the cluster.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*
COPY infra/flink/requirements.txt /opt/fraud/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /opt/fraud/requirements.txt
ADD https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.2.0-1.19/flink-sql-connector-kafka-3.2.0-1.19.jar /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar
RUN chmod 0444 /opt/flink/lib/flink-sql-connector-kafka-3.2.0-1.19.jar
COPY src /opt/fraud/src
COPY schemas /opt/fraud/schemas
ENV PYTHONPATH=/opt/fraud
# Docker seeds a new named volume from the image path it is mounted over, so the
# checkpoint directory must exist here and be owned by the flink user. Otherwise
# the volume is created root-owned and the JobMaster fails to start with
# "Failed to create directory for shared state".
RUN mkdir -p /opt/flink/checkpoints && chown -R flink:flink /opt/flink/checkpoints
USER flink
