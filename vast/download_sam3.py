from huggingface_hub import snapshot_download
print("Downloading facebook/sam3...")
snapshot_download("facebook/sam3")
print("SAM3_DOWNLOAD_DONE")
