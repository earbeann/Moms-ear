import openai
import requests
from PIL import Image
from io import BytesIO


def generate_image_from_paragraph(paragraph, style_prompt=None):
    """문단을 바탕으로 이미지를 생성하는 함수"""
    prompt = f"Create an illustration for the following story segment:\n{paragraph}"

    if style_prompt:
        prompt += f"\nIn the style of: {style_prompt}"  # 스타일 추가

    # OpenAI API 호출하여 이미지 생성
    response = openai.Image.create(
        model="dall-e-3",  # 또는 gpt-4 사용 가능
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality = "standard"
    )

    # 이미지 URL 추출
    image_url = response['data'][0]['url']

    # 이미지 다운로드 및 PIL로 열기
    image_response = requests.get(image_url)
    image = Image.open(BytesIO(image_response.content))

    return image


def save_image(image, filename):
    """생성된 이미지를 파일로 저장"""
    image.save(filename)
