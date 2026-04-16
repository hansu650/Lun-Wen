from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/swin-base-patch4-window7-224",
    local_dir="./pretrained/swin_base"
)

snapshot_download(
    repo_id="facebook/dinov2-base",
    local_dir="./pretrained/dinov2_base"
)