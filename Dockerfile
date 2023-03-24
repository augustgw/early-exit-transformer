FROM pytorch/pytorch

RUN pip install joblib
RUN pip install torchaudio

COPY . ./