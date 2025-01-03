import openai
import re

# OpenAI API 키 설정
openai.api_key = "your api_key"  # 본인의 API 키를 입력하세요.

def create_fairy_tale_from_diary(diary_content):
    """일기 내용을 바탕으로 전체 동화를 생성하는 함수"""
    prompt = (
        f"아래의 일기 내용을 바탕으로 어린이용 동화를 만들어주세요:\n"
        f"일기 내용: {diary_content}\n\n"
        f"동화 형식으로 재미있고 교훈적인 이야기로 만들어주세요."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",  # 또는 gpt-4 사용 가능
        messages=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message['content']  # 수정: 'choices' 접근 방식

def split_fairy_tale_into_paragraphs(fairy_tale):
    """생성된 동화를 문단 단위로 분할하는 함수"""
    paragraphs = re.split(r'\n+', fairy_tale.strip())  # 빈 줄 기준으로 분리
    return [p for p in paragraphs if p.strip()]  # 빈 문단 제거