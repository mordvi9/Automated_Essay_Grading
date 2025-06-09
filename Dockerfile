FROM continuumio/miniconda3:latest

# ---- create the env ----
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
 && conda clean -afy

# activate in subsequent layers
SHELL ["bash", "-c"]
ENV PATH=/opt/conda/envs/aes_project/bin:$PATH
RUN echo "source activate aes_project" >> ~/.bashrc

# ---- copy your Streamlit code ----
WORKDIR /app
COPY . /app

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
