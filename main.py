import base64
from io import BytesIO
import os
import uuid
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
import requests
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, FileResponse
import aiohttp
from validations import checkFaceSimilarity, checkfacevalidation
from pydantic import BaseModel
import io
from PIL import Image
from typing import List

app = FastAPI()

current_directory = os.path.dirname(__file__)

UPLOAD_DIR = f"{current_directory}/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


class Base64Image(BaseModel):
    base64_string: str

async def fetch_image_from_url(url: str) -> io.BytesIO:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail="Invalid URL")
            return BytesIO(await response.read())

def decode_base64_image(base64_string: str) -> BytesIO:
    image_data = base64.b64decode(base64_string)
    return BytesIO(image_data)

@app.post("/image-to-base64/")
async def image_to_base64(file: UploadFile = File(None), url: str = Form(None)):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="No image file or URL provided")

    if file:
        contents = await file.read()
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(contents)
    else:
        image_io = await fetch_image_from_url(url)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    os.remove(image_path)

    return encoded_string

@app.post("/base64-to-image/")
async def base64_to_image(base64_image: Base64Image):
    try:
        base64_string = base64_image.base64_string
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        image.save(image_path)

        return StreamingResponse(open(image_path, "rb"), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")

# Initialize global variables
upsampler = None
face_enhancer = None

def load_models_esrgan():
    global upsampler, face_enhancer

    model_path = f"{current_directory}/weights/RealESRGAN_x4plus.pth"
    netscale = 4
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    tile = 64
    tile_pad = 10
    pre_pad = 0
    outscale = 1.5

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=False,
        gpu_id=0)

    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)

async def upscale_image(image_path):
    if upsampler is None or face_enhancer is None:
        load_models_esrgan()

    imgname, extension = os.path.splitext(os.path.basename(image_path))

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None

    print("checkpoint1")
    try:
        output, _ = upsampler.enhance(img, outscale=1.5)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        extension = extension[1:]
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'

        output_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"

        cv2.imwrite(output_path, output)
        print("output:", output_path)
        return output_path

@app.post("/generate-upscaled-image/")
async def upscale_endpoint(base64_image: Base64Image):
    try:
        image_io = decode_base64_image(base64_image.base64_string)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

        upscaled_image_path = await upscale_image(image_path)

        with open(upscaled_image_path, "rb") as upscaled_image_file:
            upscaled_image_base64 = base64.b64encode(upscaled_image_file.read()).decode('utf-8')

        os.remove(image_path)
        os.remove(upscaled_image_path)

        return upscaled_image_base64
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")


@app.post("/check-face-validations/")
async def face_validation_endpoint(base64_image: Base64Image):
    try:
        image_io = decode_base64_image(base64_image.base64_string)
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(image_io.getvalue())

        is_straight, res = await checkfacevalidation(image_path)

        os.remove(image_path)

        return is_straight, res
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")



@app.post("/check-face-similarity/")
async def face_similarity_endpoint(images: List[Base64Image]):
    try:
        if len(images) != 2:
            raise HTTPException(status_code=400, detail="Two images are required")

        # Decode and save the original image
        img_1 = decode_base64_image(images[0].base64_string)
        image_1_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_1_path, "wb") as f:
            f.write(img_1.getvalue())

        # Decode and save the background image
        img_2 = decode_base64_image(images[1].base64_string)
        image_2_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_2_path, "wb") as f:
            f.write(img_2.getvalue())

        is_same = await checkFaceSimilarity(image_1_path, image_2_path)

        # Cleanup
        os.remove(image_1_path)
        os.remove(image_2_path)

        return is_same

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)