FROM python:3.9

WORKDIR /workdir

# preserving sanity
RUN echo 'alias ll="ls -l"' >> ~/.bashrc
RUN echo 'alias la="ls -la"' >> ~/.bashrc
RUN echo 'alias c="clear"' >> ~/.bashrc

RUN apt-get update && apt-get install bash bash-completion vim -y

COPY ./requirements.txt /workdir/requirements.txt
RUN pip install --no-cache-dir -r /workdir/requirements.txt \
    && rm -rf /root/.cache

COPY api/. /workdir/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]