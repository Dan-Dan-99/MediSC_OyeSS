FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install vim wget curl ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx git gcc-8 g++-8 -y && \
    apt-get clean

RUN mkdir input output


RUN git clone https://github.com/amuck667/TrackEval.git
WORKDIR TrackEval
RUN git switch devel-kp \
    mkdir data data/gt data/trackers

RUN git clone https://github.com/Dan-Dan-99/MediSC_OyeSS.git
WORKDIR ../MediSC_OyeSS
RUN git switch MMPose

RUN pip install torch==2.4.1 torchvision==2.4.1 torchaudio==0.19.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -U mmengine
RUN mim install mmcv==2.1.0 mmdet>=3.1.0
RUN pip install -r requirements.txt
RUN pip install -v -e .

RUN mv Process_SkillEval.sh /usr/local/bin/Process_SkillEval.sh

ENTRYPOINT ["tail", "-f", "/dev/null"]
# RUN ["bash", "/usr/local/bin/Process_SkillEval.sh"]