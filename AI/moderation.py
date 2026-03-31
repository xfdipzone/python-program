# coding=utf-8
from openai import OpenAI
from google.colab import userdata

"""
AI 检测文字内容是否符合法规

dependency packages
pip install openai
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

"""
判断文字内容是否被标记
例如包含色情，暴力等不合适的内容
调用 client.chat.completions.create 接口实现
"""
def custom_moderation(content, parameters):
    # Define the prompt
    prompt = f"""请评估以下内容是否存在不当内容。评估应基于给定的参数。
    您的答案应为 JSON 格式，并包含以下字段：
        - flagged: 布尔值，指示内容是否因参数中的任何类别而被标记
        - reason: 字符串，使用中文解释标记原因（如有）
        - parameters: 字典，包含用于评估的参数及其值
    Parameters: {parameters}\n\nContent:\n{content}\n\nAssessment:"""

    # Call model with the prompt
    try:
        response = client.chat.completions.create(
            model=COMPLETION_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "您是一名内容审核助理。"},
                {"role": "user", "content": prompt}
            ]
        )
    except Exception as e:
        raise e

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


"""
text : I love my motherland
result : {
  "flagged": false,
  "reason": "",
  "parameters": {
    "hate": false,
    "hate/threatening": false,
    "self-harm": false,
    "sexual": false,
    "sexual/minors": false,
    "violence": false,
    "violence/graphic": false
  }
}


text : I'm going to rape you daughter and then chop you to death with a knife
result : {
  "flagged": true,
  "reason": "内容包含极端暴力和性侵犯的威胁，违反了hate/threatening和violence/graphic的参数。",
  "parameters": {
    "hate": true,
    "hate/threatening": true,
    "self-harm": false,
    "sexual": true,
    "sexual/minors": false,
    "violence": true,
    "violence/graphic": true
  }
}
"""
