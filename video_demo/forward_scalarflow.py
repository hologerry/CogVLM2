import io
import os

import numpy as np
import torch

from decord import VideoReader, bridge, cpu
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_video(video_data, strategy="chat"):
    bridge.set_bridge("torch")
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    if strategy == "base":
        clip_end_sec = 60
        clip_start_sec = 0
        start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
        end_frame = (
            min(total_frames, int(clip_end_sec * decord_vr.get_avg_fps()))
            if clip_end_sec is not None
            else total_frames
        )
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    elif strategy == "chat":
        timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
        timestamps = [i[0] for i in timestamps]
        max_second = round(max(timestamps)) + 1
        frame_id_list = []
        for second in range(max_second):
            closest_num = min(timestamps, key=lambda x: abs(x - second))
            index = timestamps.index(closest_num)
            frame_id_list.append(index)
            if len(frame_id_list) >= num_frames:
                break

        # while len(frame_id_list) < num_frames:
        #     frame_id_list.append(frame_id_list[-1])

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


def predict(video_path, model, tokenizer, temperature=0.1, torch_type=torch.bfloat16, device="cuda"):
    strategy = "chat"
    prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects. Based on your observations describe this video in detail"

    with open(video_path, "rb") as f:
        video_data = f.read()

    video = load_video(video_data, strategy=strategy)

    history = []
    query = prompt
    inputs = model.build_conversation_input_ids(
        tokenizer=tokenizer, query=query, images=[video], history=history, template_version=strategy
    )
    inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(0).to(device),
        "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(device),
        "attention_mask": inputs["attention_mask"].unsqueeze(0).to(device),
        "images": [[inputs["images"][0].to(device).to(torch_type)]],
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,
        "top_k": 1,
        "do_sample": True,
        "top_p": 0.1,
        "temperature": temperature,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def main(
    device_id: int = 0,
    job_idx: int = 0,
    num_jobs: int = 4,
    skip_existing: bool = True,
    scalarflow_data_root: str = "/data/Dynamics/ScalarFlow_cogvideox_dataset",
):
    print(f"device_id: {device_id}, job_idx: {job_idx}, num_jobs: {num_jobs}")
    if skip_existing:
        print("skip_existing is set to True, skipping existing files")

    videos_folder = os.path.join(scalarflow_data_root, "videos")
    assert os.path.exists(videos_folder), f"videos_folder {videos_folder} does not exist"

    video_names = os.listdir(videos_folder)
    cur_job_video_names = video_names[job_idx::num_jobs]
    cur_job_video_files = [os.path.join(videos_folder, video_name) for video_name in cur_job_video_names]

    labels_folder = os.path.join(scalarflow_data_root, "labels")
    os.makedirs(labels_folder, exist_ok=True)

    model_path = "THUDM/cogvlm2-video-llama3-chat"

    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    torch_type = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = (
        AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_type, trust_remote_code=True)
        .eval()
        .to(device)
    )

    for video_path in tqdm(cur_job_video_files, desc=f"Processing videos device {device_id} job {job_idx}"):
        label_name = os.path.basename(video_path).replace(".mp4", ".txt")
        response_txt_path = f"{labels_folder}/{label_name}"
        if skip_existing and os.path.exists(response_txt_path):
            continue
        response = predict(video_path, model, tokenizer, temperature=0.1, torch_type=torch_type, device=device)
        with open(response_txt_path, "w") as f:
            f.write(response)


if __name__ == "__main__":
    Fire(main)
