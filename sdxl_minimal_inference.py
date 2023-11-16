# 手元で推論を行うための最低限のコード。HuggingFace／DiffusersのCLIP、schedulerとVAEを使う
# Minimal code for performing inference at local. Use HuggingFace/Diffusers CLIP, scheduler and VAE

import argparse
import datetime
import math
import os
import random
import sys
import io
from einops import repeat
import numpy as np
import requests
import json
import time
import re
import torch
from contextlib import contextmanager
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
from tqdm import tqdm
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler
from PIL import Image
import open_clip
from safetensors.torch import load_file

from library import model_util, sdxl_model_util
import networks.lora as lora

# scheduler: このあたりの設定はSD1/2と同じでいいらしい
# scheduler: The settings around here seem to be the same as SD1/2
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


# Time EmbeddingはDiffusersからのコピー
# Time Embedding is copied from Diffusers


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device
        )
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding

def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    # x = rearrange(x, "b d -> (b d)")
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    # emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb

if __name__ == "__main__":
    # 画像生成条件を変更する場合はここを変更 / change here to change image generation conditions

    # SDXLの追加のvector embeddingへ渡す値 / Values to pass to additional vector embedding of SDXL
    target_height = 1024
    target_width = 1024
    original_height = target_height
    original_width = target_width
    crop_top = 0
    crop_left = 0

    steps = 50
    guidance_scale = 7
    seed = None  # 1

    DEVICE = "cuda"
    DTYPE = torch.float16  # bfloat16 may work

    getJobURL = os.environ.get("HELIX_NEXT_TASK_URL", "")
    readSessionURL = os.environ.get("HELIX_INITIAL_SESSION_URL", "")
    
    # this points at the axolotl or sd-scripts repo in a relative way
    # to where the helix runner is active
    appFolder = os.environ.get("APP_FOLDER", "")

    if getJobURL == "":
        sys.exit("HELIX_GET_JOB_URL is not set")

    if readSessionURL == "":
        sys.exit("HELIX_GET_SESSION_URL is not set")

    if appFolder == "":
        sys.exit("APP_FOLDER is not set")

    print(f"🟡 HELIX_NEXT_TASK_URL {getJobURL} --------------------------------------------------\n")
    print(f"🟡 HELIX_INITIAL_SESSION_URL {readSessionURL} --------------------------------------------------\n")
    print(f"🟡 APP_FOLDER {appFolder} --------------------------------------------------\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--prompt2", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=".")

    # these weights will be loaded from the initial session
    # parser.add_argument(
    #     "--lora_weights",
    #     type=str,
    #     nargs="*",
    #     default=[],
    #     help="LoRA weights, only supports networks.lora, each argument is a `path;multiplier` (semi-colon separated)",
    # )
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.prompt2 is None:
        args.prompt2 = args.prompt

    # HuggingFaceのmodel id
    text_encoder_1_name = "openai/clip-vit-large-patch14"
    text_encoder_2_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    # checkpointを読み込む。モデル変換についてはそちらの関数を参照
    # Load checkpoint. For model conversion, see this function

    # 本体RAMが少ない場合はGPUにロードするといいかも
    # If the main RAM is small, it may be better to load it on the GPU
    text_model1, text_model2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
        sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, args.ckpt_path, "cpu"
    )

    # Text Encoder 1はSDXL本体でもHuggingFaceのものを使っている
    # In SDXL, Text Encoder 1 is also using HuggingFace's

    # Text Encoder 2はSDXL本体ではopen_clipを使っている
    # それを使ってもいいが、SD2のDiffusers版に合わせる形で、HuggingFaceのものを使う
    # 重みの変換コードはSD2とほぼ同じ
    # In SDXL, Text Encoder 2 is using open_clip
    # It's okay to use it, but to match the Diffusers version of SD2, use HuggingFace's
    # The weight conversion code is almost the same as SD2

    # VAEの構造はSDXLもSD1/2と同じだが、重みは異なるようだ。何より謎のscale値が違う
    # fp16でNaNが出やすいようだ
    # The structure of VAE is the same as SD1/2, but the weights seem to be different. Above all, the mysterious scale value is different.
    # NaN seems to be more likely to occur in fp16

    unet.to(DEVICE, dtype=DTYPE)
    unet.eval()

    vae_dtype = DTYPE
    if DTYPE == torch.float16:
        print("use float32 for vae")
        vae_dtype = torch.float32
    vae.to(DEVICE, dtype=vae_dtype)
    vae.eval()

    text_model1.to(DEVICE, dtype=DTYPE)
    text_model1.eval()
    text_model2.to(DEVICE, dtype=DTYPE)
    text_model2.eval()

    unet.set_use_memory_efficient_attention(True, False)
    if torch.__version__ >= "2.0.0": # PyTorch 2.0.0 以上対応のxformersなら以下が使える
        vae.set_use_memory_efficient_attention_xformers(True)

    # Tokenizers
    tokenizer1 = CLIPTokenizer.from_pretrained(text_encoder_1_name)
    tokenizer2 = lambda x: open_clip.tokenize(x, context_length=77)

    lora_weights = []
    waiting_for_initial_session = True

    # we need to load the first task to know what the Lora weights are
    # perhaps there are no lora weights in which case we will skip
    # this step - we are not popping the task from the queue
    # rather waiting until it appears so we can know what lora weights to
    # load (if any)
    while waiting_for_initial_session:
        response = requests.get(readSessionURL)
        if response.status_code != 200:
            time.sleep(0.1)
            continue
        
        session = json.loads(response.content)
        waiting_for_initial_session = False
        if session["finetune_file"] != "":
            lora_weights = [session["finetune_file"]]

    print("🟡 Lora weights --------------------------------------------------\n")
    print(lora_weights)

    # LoRA
    for weights_file in lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 0.7

        lora_model, weights_sd = lora.create_network_from_weights(
            multiplier, weights_file, vae, [text_model1, text_model2], unet, None, True
        )
        lora_model.merge_to([text_model1, text_model2], unet, weights_sd, DTYPE, DEVICE)

    # scheduler
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
    )

    def generate_image(session_id, prompt, prompt2, negative_prompt, seed=None):
        # 将来的にサイズ情報も変えられるようにする / Make it possible to change the size information in the future
        # prepare embedding
        with torch.no_grad():
            # vector
            emb1 = get_timestep_embedding(torch.FloatTensor([original_height, original_width]).unsqueeze(0), 256)
            emb2 = get_timestep_embedding(torch.FloatTensor([crop_top, crop_left]).unsqueeze(0), 256)
            emb3 = get_timestep_embedding(torch.FloatTensor([target_height, target_width]).unsqueeze(0), 256)
            # print("emb1", emb1.shape)
            c_vector = torch.cat([emb1, emb2, emb3], dim=1).to(DEVICE, dtype=DTYPE)
            uc_vector = c_vector.clone().to(DEVICE, dtype=DTYPE)  # ちょっとここ正しいかどうかわからない I'm not sure if this is right

            # crossattn

        # Text Encoderを二つ呼ぶ関数  Function to call two Text Encoders
        def call_text_encoder(text, text2):
            # text encoder 1
            batch_encoding = tokenizer1(
                text,
                truncation=True,
                return_length=True,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            tokens = batch_encoding["input_ids"].to(DEVICE)

            with torch.no_grad():
                enc_out = text_model1(tokens, output_hidden_states=True, return_dict=True)
                text_embedding1 = enc_out["hidden_states"][11]
                # text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)    # layer normは通さないらしい

            # text encoder 2
            with torch.no_grad():
                tokens = tokenizer2(text2).to(DEVICE)

                enc_out = text_model2(tokens, output_hidden_states=True, return_dict=True)
                text_embedding2_penu = enc_out["hidden_states"][-2]
                # print("hidden_states2", text_embedding2_penu.shape)
                text_embedding2_pool = enc_out["text_embeds"]   # do not support Textual Inversion

            # 連結して終了 concat and finish
            text_embedding = torch.cat([text_embedding1, text_embedding2_penu], dim=2)
            return text_embedding, text_embedding2_pool

        # cond
        c_ctx, c_ctx_pool = call_text_encoder(prompt, prompt2)
        # print(c_ctx.shape, c_ctx_p.shape, c_vector.shape)
        c_vector = torch.cat([c_ctx_pool, c_vector], dim=1)

        # uncond
        uc_ctx, uc_ctx_pool = call_text_encoder(negative_prompt, negative_prompt)
        uc_vector = torch.cat([uc_ctx_pool, uc_vector], dim=1)

        text_embeddings = torch.cat([uc_ctx, c_ctx])
        vector_embeddings = torch.cat([uc_vector, c_vector])

        # メモリ使用量を減らすにはここでText Encoderを削除するかCPUへ移動する

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # # random generator for initial noise
            # generator = torch.Generator(device="cuda").manual_seed(seed)
            generator = None
        else:
            generator = None

        # get the initial random noise unless the user supplied it
        # SDXLはCPUでlatentsを作成しているので一応合わせておく、Diffusersはtarget deviceでlatentsを作成している
        # SDXL creates latents in CPU, Diffusers creates latents in target device
        latents_shape = (1, 4, target_height // 8, target_width // 8)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(DEVICE, dtype=DTYPE)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * scheduler.init_noise_sigma

        # set timesteps
        scheduler.set_timesteps(steps, DEVICE)

        # このへんはDiffusersからのコピペ
        # Copy from Diffusers
        timesteps = scheduler.timesteps.to(DEVICE)  # .to(DTYPE)
        num_latent_input = 2
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                noise_pred = unet(latent_model_input, t, text_embeddings, vector_embeddings)

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            # latents = 1 / 0.18215 * latents
            latents = 1 / sdxl_model_util.VAE_SCALE_FACTOR * latents
            latents = latents.to(vae_dtype)
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # image = self.numpy_to_pil(image)
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]

        # 保存して終了 save and finish
        image_paths = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for i, img in enumerate(image):
            image_path = os.path.join(appFolder, args.output_dir, f"image_{session_id}_{timestamp}_{i:03d}.png")
            image_paths.append(image_path)
            img.save(image_path)
        return image_paths

    while True:
        response = requests.get(getJobURL)
        if response.status_code != 200:
            time.sleep(0.1)
            continue

        last_seen_progress = 0

        task = json.loads(response.content)
        session_id = task["session_id"]

        print("🟡 SDXL Job --------------------------------------------------\n")
        print(task)

        print(f"[SESSION_START]session_id={session_id}", file=sys.stdout)

        image_paths = generate_image(task["session_id"], task["prompt"], "", "", seed)

        print(f"[SESSION_END]{json.dumps(image_paths)}", file=sys.stdout)
        print("🟡 SDXL Result --------------------------------------------------\n")
        print(image_paths)
