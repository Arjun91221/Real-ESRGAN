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

app = FastAPI()

current_directory = os.path.dirname(__file__)

UPLOAD_DIR = f"{current_directory}/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

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
    outscale = 2.5

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

def upscale_image(image_path):
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

        _, buffer = cv2.imencode(f'.{extension}', output)
        image_io = BytesIO(buffer)
        image_io.seek(0)
        return image_io


        cv2.imwrite(output_path, output)
        print("output:", output_path)
        return output_path



@app.post("/generate-upscaled-image/")
async def upscale_endpoint(file: UploadFile = File(None), url: str = Form(None)):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="No image provided. Provide either a file or a URL.")

    if file:
        contents = await file.read()
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(contents)
    elif url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail="Unable to download image from URL")
                    contents = await response.read()
                    image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
                    with open(image_path, "wb") as f:
                        f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

    upscale_image_io = await upscale_image(image_path)
    if upscale_image_io is None:
        raise HTTPException(status_code=500, detail="Error in upscaling image")


    os.remove(image_path)

    return StreamingResponse(upscale_image_io, media_type="image/png")


@app.post("/check-face-validations/")
async def face_validation_endpoint(file: UploadFile = File(None), url: str = Form(None)):
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="No image provided. Provide either a file or a URL.")

    if file:
        contents = await file.read()
        image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
        with open(image_path, "wb") as f:
            f.write(contents)
    elif url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail="Unable to download image from URL")
                    contents = await response.read()
                    image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
                    with open(image_path, "wb") as f:
                        f.write(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

    is_straight, res = await checkfacevalidation(image_path)

    os.remove(image_path)

    return is_straight, res


@app.post("/check-face-similarity/")
async def face_similarity_endpoint(file1: UploadFile = File(None), file2: UploadFile = File(None), url1: str = Form(None), url2: str = Form(None)):
    if (file1 is None or file2 is None) and (url1 is None or url2 is None):
        raise HTTPException(status_code=400, detail="Provide either two files or two URLs.")

    img1_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"
    img2_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"

    if file1 and file2:
        contents1 = await file1.read()
        with open(img1_path, "wb") as f:
            f.write(contents1)

        contents2 = await file2.read()
        with open(img2_path, "wb") as f:
            f.write(contents2)

    elif url1 and url2:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url1) as response1:
                    if response1.status != 200:
                        raise HTTPException(status_code=400, detail="Unable to download image 1 from URL")
                    contents1 = await response1.read()
                    with open(img1_path, "wb") as f:
                        f.write(contents1)

                async with session.get(url2) as response2:
                    if response2.status != 200:
                        raise HTTPException(status_code=400, detail="Unable to download image 2 from URL")
                    contents2 = await response2.read()
                    with open(img2_path, "wb") as f:
                        f.write(contents2)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error downloading images: {str(e)}")

    try:
        is_same = await checkFaceSimilarity(img1_path, img2_path)
    finally:
        os.remove(img1_path)
        os.remove(img2_path)

    return is_same

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)