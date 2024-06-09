from PIL import Image, ImageDraw


def render_page(page, width, height):
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for box in page:
        if len(box) == 4:
            draw.polygon(box, outline=(0, 0, 0))
        elif len(box) == 8:
            _box = [(box[i], box[i + 1]) for i in range(0, 8, 2)]
            draw.polygon(_box, outline=(0, 0, 0))
    return canvas
