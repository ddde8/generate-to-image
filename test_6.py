from PIL import Image

im = Image.open("/Users/jieunchoi/Documents/GitHub/generate-to-image/123.jpeg").convert("RGB")
im.save("123_cleaned.jpg")
