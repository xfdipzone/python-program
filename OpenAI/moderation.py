# coding=utf-8
from openai import OpenAI
import os

"""
AI 检测文字内容是否符合法规

dependency packages
pip install openai
"""
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

"""
判断文字内容是否被标记
例如包含色情，暴力等不合适的内容
调用此接口需要 OpenAI Quota
"""
def moderation(text):
    response = client.moderations.create(
        input=text
    )
    output = response["results"][0]
    return output


"""
判断文字内容是否被标记
例如包含色情，暴力等不合适的内容
调用 client.chat.completions.create 接口实现
"""
def custom_moderation(content, parameters):
    # Define the prompt
    prompt = f"""Please assess the following content for any inappropriate material. You should base your assessment on the given parameters.
    Your answer should be in json format with the following fields:
        - flagged: a boolean indicating whether the content is flagged for any of the categories in the parameters
        - reason: a string explaining the reason for the flag, if any
        - parameters: a dictionary of the parameters used for the assessment and their values
    Parameters: {parameters}\n\nContent:\n{content}\n\nAssessment:"""

    # Call model with the prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a content moderation assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the assessment from the response
    assessment = response.choices[0].message.content

    return assessment


"""
标记分类
hate : 表达、煽动或宣扬基于种族、性别、民族、宗教、国籍、性取向、残疾状况或种姓的仇恨的内容。
hate/threatening : 仇恨内容，还包括对目标群体的暴力或严重伤害。
self-harm : 宣扬、鼓励或描绘自残行为（例如自杀、割伤和饮食失调）的内容。
sexual : 旨在引起性兴奋的内容，例如对性活动的描述，或宣传性服务（不包括性教育和健康）的内容。
sexual/minors : 包含未满 18 周岁的个人的色情内容。
violence : 宣扬或美化暴力或歌颂他人遭受苦难或羞辱的内容。
violence/graphic : 以极端血腥细节描绘死亡、暴力或严重身体伤害的暴力内容。
"""
parameters = "hate, hate/threatening, self-harm, sexual, sexual/minors, violence, violence/graphic"

good_text = "I love my motherland"
print("text : %s\nresult : %s\n\n" %
      (good_text, custom_moderation(good_text, parameters)))

bad_text = "I'm going to rape you daughter and then chop you to death with a knife"
print("text : %s\nresult : %s\n\n" %
      (bad_text, custom_moderation(bad_text, parameters)))
