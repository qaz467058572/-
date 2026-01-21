import discord
from openai import AsyncOpenAI
import asyncio
import re
import requests
import json
# -*- coding: utf-8 -*-

# 配置OpenAI客户端（本地LM Studio/Ollama）
client = AsyncOpenAI(
    api_key="any-string",  # 本地服务无需真实Key
    base_url="http://198.18.0.1:1234/v1"  # LM Studio默认端口，Ollama改为http://localhost:11434/v1
)

async def get_llm_response(messages):
    """调用LLM获取回复（传入结构化历史：role+昵称+内容）"""
    try:
        response = await client.chat.completions.create(
            model="qwen3-4b",  # 替换为本地加载的模型名（必须完全匹配）
            messages=messages,
            max_tokens=10000,
            temperature=0.7,
            extra_body={"context_window": 8192}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_detail = f"调用失败：类型={type(e).__name__}，信息={str(e)}"
        print(error_detail)
        return error_detail

async def summarize_conversation(messages):
    """生成对话总结（强制保留发言者昵称）"""
    # 构造总结提示词：明确要求保留昵称
    summary_prompt = [
        {
            "role": "system",
            "content": """请总结以下群组对话内容，严格要求：
1. 必须保留每个发言者的昵称（格式：昵称：内容）；
2. 提炼核心问题、回复和结论，不遗漏关键信息；
3. 简洁明了，不超过200字；
4. 纯文本总结，无额外格式。"""
        },
        {
            "role": "user",
            "content": f"需要总结的对话：{json.dumps(messages, ensure_ascii=False, indent=2)}"
        }
    ]
    try:
        response = await client.chat.completions.create(
            model="qwen3-4b",
            messages=summary_prompt,
            max_tokens=300,
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        return {"role": "system", "content": summary}
    except Exception as e:
        print(f"总结生成失败：{str(e)}")
        return {"role": "system", "content": "【对话总结】：此前对话内容较多，核心信息未完全提取，继续当前对话。"}

async def clean_tag_content(response, tag="think"):
    """移除模型输出的冗余标签（如）"""
    patterns = [
        rf'^.*?</{tag}>',
        rf'<{tag}>.*?</{tag}>',
        rf'<{tag.upper()}>.*?</{tag.upper()}>',
        rf'{{{tag}}}.*?{{/{tag}}}',
        rf'\[{tag}\].*?\[/{tag}\]',
        rf'{{{tag}：}}.*?{{/{tag}}}',
    ]
    combined_pattern = re.compile('|'.join(patterns), re.DOTALL | re.IGNORECASE)
    cleaned_text = combined_pattern.sub('', response).strip()
    return cleaned_text if cleaned_text else "我已收到你的消息啦～"