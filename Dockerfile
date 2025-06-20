FROM continuumio/miniconda3:latest
WORKDIR /app


# 1) Copy the env spec & build your conda env
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

ENV PATH=/opt/conda/envs/aes_project/bin:$PATH

# 2) spaCy model
RUN conda run -n aes_project python -m spacy download en_core_web_sm

# ← Add this to bake in your local artifacts:
COPY models/ /app/models/

# 3) Copy your code
COPY Pipeline.py /app/Pipeline.py  # if you added the shim
COPY src/ /app/src/

EXPOSE 8501
ENTRYPOINT ["bash","-lc","conda run -n aes_project --no-capture-output streamlit run src/app.py --server.port ${PORT} --server.address 0.0.0.0"]
