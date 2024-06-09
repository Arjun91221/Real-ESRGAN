from io import BytesIO
import os
import uuid
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
import requests
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

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
    if image_path.startswith("http://") or image_path.startswith("https://"):
        response = requests.get(image_path)

        if response.status_code == 200:
            # Save the image to a temporary file
            temp_file_path = str(uuid.uuid4()) + ".png"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(response.content)
        else:
            print(f"Failed to download the image. Status code: {response.status_code}")
            return
    else:
        # Assume the input is a local file path
        temp_file_path = image_path

    if upsampler is None or face_enhancer is None:
        load_models_esrgan()

    imgname, extension = os.path.splitext(os.path.basename(temp_file_path))
    print('Testing', imgname)

    img = cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED)

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
async def upscale_endpoint(file: UploadFile = File(...)):
    # if file.content_type not in ["image/jpeg", "image/png"]:
    #     raise HTTPException(status_code=400, detail="Invalid file type")


    contents = await file.read()
    image_path = f"{UPLOAD_DIR}/{str(uuid.uuid4())}.png"

    with open(image_path, "wb") as f:
        f.write(contents)

    upscale_image_io = upscale_image(image_path)
    if upscale_image_io is None:
        raise HTTPException(status_code=500, detail="Error in upscaling image")


    os.remove(image_path)

    return StreamingResponse(upscale_image_io, media_type="image/png")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cropped_path = auto_crop(file_path)
    if cropped_path is None:
        raise HTTPException(status_code=500, detail="Error in cropping image")

    return FileResponse(cropped_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)