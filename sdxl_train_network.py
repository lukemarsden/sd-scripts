import argparse
import torch
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
import tempfile
import os
import io
import sys
import time
from pathlib import Path
import time
import requests
import datetime
import json
import re
from library import sdxl_model_util, sdxl_train_util, train_util
from contextlib import contextmanager
import train_network

toml_template = '''
# for sdxl fine tuning

[general]
enable_bucket = true                        # Whether to use Aspect Ratio Bucketing

[[datasets]]
resolution = 1024                           # Training resolution
batch_size = 4                              # Batch size

  [[datasets.subsets]]
  image_dir = '{dataset_path}' # Specify the folder containing the training images
  caption_extension = '.txt'                # Caption file extension; change this if using .txt
  num_repeats = 10                          # Number of repetitions for training images

'''

class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def load_tokenizer(self, args):
        tokenizer = sdxl_train_util.load_tokenizers(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                print("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            dataset.cache_text_encoder_outputs(
                tokenizers,
                text_encoders,
                accelerator.device,
                weight_dtype,
                args.cache_text_encoder_outputs_to_disk,
                accelerator.is_main_process,
            )

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not args.lowram:
                print("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device)
            text_encoders[1].to(accelerator.device)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            with torch.enable_grad():
                # Get the text embedding for conditioning
                # TODO support weighted captions
                # if args.weighted_captions:
                #     encoder_hidden_states = get_weighted_text_embeddings(
                #         tokenizer,
                #         text_encoder,
                #         batch["captions"],
                #         accelerator.device,
                #         args.max_token_length // 75 if args.max_token_length else 1,
                #         clip_skip=args.clip_skip,
                #     )
                # else:
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                    args.max_token_length,
                    input_ids1,
                    input_ids2,
                    tokenizers[0],
                    tokenizers[1],
                    text_encoders[0],
                    text_encoders[1],
                    None if not args.full_fp16 else weight_dtype,
                )
        else:
            encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
            encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
            pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)

            # # verify that the text encoder outputs are correct
            # ehs1, ehs2, p2 = train_util.get_hidden_states_sdxl(
            #     args.max_token_length,
            #     batch["input_ids"].to(text_encoders[0].device),
            #     batch["input_ids2"].to(text_encoders[0].device),
            #     tokenizers[0],
            #     tokenizers[1],
            #     text_encoders[0],
            #     text_encoders[1],
            #     None if not args.full_fp16 else weight_dtype,
            # )
            # b_size = encoder_hidden_states1.shape[0]
            # assert ((encoder_hidden_states1.to("cpu") - ehs1.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((encoder_hidden_states2.to("cpu") - ehs2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # assert ((pool2.to("cpu") - p2.to(dtype=weight_dtype)).abs().max() > 1e-2).sum() <= b_size * 2
            # print("text encoder outputs verified")

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser

def extract_percentage(text):
    match = re.search(r'steps:\s+(\d+)%', text)
    if match:
        return int(match.group(1))
    else:
        return None

@contextmanager
def redirect_stderr_to_function(func, buffer_size=1024, url="", sessionid=""):
    class BufferedBytesStream(io.BytesIO):
        def __init__(self, buffer_size):
            super().__init__()
            self.buffer_size = buffer_size
            self.buffer = bytearray()

        def write(self, b):
            if isinstance(b, str):
                b = b.encode('utf-8')
            self.buffer.extend(b)
            while len(self.buffer) >= self.buffer_size:
                chunk, self.buffer = self.buffer[:self.buffer_size], self.buffer[self.buffer_size:]
                # this is capture_model_output_chunk
                func(url, sessionid, chunk)

    original_stderr = sys.stderr
    sys.stderr = BufferedBytesStream(buffer_size)

    try:
        yield
    finally:
        # Flush remaining bytes in buffer, if any
        if len(sys.stderr.buffer) > 0:
            # this is capture_model_output_chunk
            func(url, sessionid, sys.stderr.buffer)
        sys.stderr = original_stderr

last_seen_percent = 0

def capture_model_output_chunk(url, session_id, b):
    global last_seen_percent
    message = b.decode('utf-8')
    percent = extract_percentage(message)
    if percent is not None and percent is not last_seen_percent:
        last_seen_percent = percent
        print(f"percent: {percent}")
        json_payload = json.dumps({
            "type": "progress",
            "session_id": session_id,
            "progress": percent,
        })
        requests.post(url, data=json_payload)

if __name__ == "__main__":
    getJobURL = os.environ.get("HELIX_GET_JOB_URL", None)
    respondJobURL = os.environ.get("HELIX_RESPOND_JOB_URL", None)
    appFolder = os.environ.get("APP_FOLDER", None)

    if getJobURL is None:
        sys.exit("HELIX_GET_JOB_URL is not set")

    if respondJobURL is None:
        sys.exit("HELIX_RESPOND_JOB_URL is not set")

    if appFolder is None:
        sys.exit("APP_FOLDER is not set")

    parser = setup_parser()
    cliArgs = parser.parse_args()

    waitLoops = 0
    
    while True:
        response = requests.get(getJobURL)
        if response.status_code != 200:
            time.sleep(0.1)
            waitLoops = waitLoops + 1
            if waitLoops % 10 == 0:
                print("--------------------------------------------------\n")
                current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"{current_timestamp} waiting for next job {getJobURL} {response.status_code}")
            continue

        waitLoops = 0
        last_seen_progress = 0

        task = json.loads(response.content)

        print("🟡 SDXL Finetine Job --------------------------------------------------\n")
        print(task)

        session_id = task["session_id"]
        finetune_input_dir = task["finetune_input_dir"]

        results_dir = f"/tmp/helix/results/{session_id}"
        
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as temp:
            config_path = temp.name
        
        values = {
            'dataset_path': finetune_input_dir
        }

        filled_template = toml_template.format(**values)
        with open(config_path, 'w') as f:
            f.write(filled_template)

        print("🟡 SDXL Config File --------------------------------------------------\n")
        print(config_path)

        print("🟡 SDXL Config --------------------------------------------------\n")
        print(filled_template)

        print("🟡 SDXL Inputs --------------------------------------------------\n")
        print(finetune_input_dir)

        print("🟡 SDXL Outputs --------------------------------------------------\n")
        print(results_dir)

        cliArgs.dataset_config = config_path
        cliArgs.output_dir = results_dir

        args = train_util.read_config_from_file(cliArgs, parser)

        with redirect_stderr_to_function(capture_model_output_chunk, buffer_size=20, url=respondJobURL, sessionid=session_id):
            trainer = SdxlNetworkTrainer()
            trainer.train(args)

        final_file = results_dir + "/lora.safetensors"

        print("🟡 SDXL Result --------------------------------------------------\n")
        print(final_file)
        json_payload = json.dumps({
            "type": "result",
            "session_id": task["session_id"],
            "files": [final_file]
        })
        requests.post(respondJobURL, data=json_payload)

        time.sleep(1)
        sys.exit(0)



    
