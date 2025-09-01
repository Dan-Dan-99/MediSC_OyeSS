FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install vim wget curl ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx git -y && \
    apt-get clean

RUN mkdir input output

RUN git clone -b devel-kp https://github.com/amuck667/TrackEval.git
RUN git clone -b MMPose https://github.com/Dan-Dan-99/MediSC_OyeSS.git

WORKDIR TrackEval
RUN mkdir data data/gt data/trackers

WORKDIR ../MediSC_OyeSS
RUN pip install -U torch==2.4.1 torchaudio==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install openmim
RUN mim install -U mmengine mmcv==2.1.0 mmdet>=3.1.0
RUN pip install -r requirements.txt

RUN pip install gdown
ARG GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/16bQw6r2m9EUsMAh5wwy8d9GuM4NL9BSI?usp=sharing"
RUN gdown --folder --fuzzy "$GDRIVE_FOLDER_URL" -O /workspace/MediSC_OyeSS/configs

RUN mv Process_skillEval.sh /usr/local/bin/Process_skillEval.sh
RUN chmod +777 /usr/local/bin/Process_skillEval.sh

ENTRYPOINT ["tail", "-f", "/dev/null"]
#CMD ["bash"]
