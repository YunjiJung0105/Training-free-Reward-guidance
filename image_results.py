from PIL import Image

# 8개의 이미지 파일 경로
file_dir = "forward_recurr_1_scale3/0"
image_paths = [
    "0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png",
]

# 이미지 불러오기
images = [Image.open(f'{file_dir}/{path}') for path in image_paths]


# 전체 연결된 이미지의 너비와 높이 계산
image_width, image_height = images[0].size
total_width = image_width * len(images)
total_height = image_height

# 새로운 캔버스 생성
result_image = Image.new("RGB", (total_width, total_height))

# 이미지를 한 줄로 배치
x_offset = 0
for img in images:
    result_image.paste(img, (x_offset, 0))
    x_offset += img.width

# 결과 이미지 저장
result_image.save(f"{file_dir}/concatenated_image.jpg")
