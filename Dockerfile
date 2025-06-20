FROM continuumio/miniconda3:latest
WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

ENV PATH=/opt/conda/envs/aes_project/bin:$PATH
RUN conda run -n aes_project python -m spacy download en_core_web_sm

COPY . .

RUN conda install -n aes_project -c pytorch pytorch=2.5 torchvision=0.20 torchaudio=2.5 --yes && conda clean -afy

ENV PORT=8501

EXPOSE 8501

# Use bash -lc so $PORT is expanded (and defaults to 8501 if unset)
ENTRYPOINT ["bash","-lc","conda run -n aes_project --no-capture-output streamlit run src/app.py --server.port ${PORT} --server.address 0.0.0.0"]