
import os
from openai import OpenAI

def test_openai_api():
    """
    测试OpenAI API是否能正常连接和使用
    @return 如果成功，返回API响应；如果失败，返回错误信息
    """
    try:
        
        # 初始化OpenAI客户端
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                                          # base_url=os.getenv("GPT_BASE_URL")
                                          
        
        # 发送简单测试请求
        response = client.chat.completions.create(
            model=os.getenv("GPT_MODEL_ID"),
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "你好，这是一个API测试。顺便告诉我你是谁？"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # 提取并返回回答
        answer = response.choices[0].message.content.strip()
        return f"API测试成功！回答：{answer}"
    
    except Exception as e:
        return f"API测试失败：{str(e)}"

if __name__ == "__main__":
    print("正在测试OpenAI API连接...")
    result = test_openai_api()
    print(result) 