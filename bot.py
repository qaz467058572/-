import discord
import responses
# -*- coding: utf-8 -*-

async def send_message(message, user_message, channel_id, user_id, user_name, is_private):
    try:
        response = await responses.get_response(channel_id, user_id, user_name, user_message)
        if is_private:
            await message.author.send(response)
        else:
            await message.channel.send(content=response, reference=message)
    except Exception as e:
        print(f"发送消息时出错：{e}")

def run_discord_bot():
    token = ''  # 请替换为自己的Token
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print('We have logged in as {0.user}'.format(client))

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        # 关键优化：用display_name获取用户当前显示的昵称（支持服务器自定义昵称）
        username = message.author.display_name
        user_message = str(message.content).strip()
        channel_id = message.channel.id
        user_id = message.author.id
        print(username,":",message.content)

        # 指令处理
        if user_message == "！清空記憶":
            await responses.clear_group_history(channel_id)
            await message.channel.send("已清空本頻道所有用戶記憶～")
            return
        if user_message == "！清空我的記憶":
            res = await responses.clear_user_history(channel_id, user_id)
            await message.channel.send("已清空用戶記憶～")
            return

        # 处理@机器人场景
        if client.user.mentioned_in(message):
            user_message = user_message.replace(client.user.mention, "").strip()
            if not user_message:
                await message.channel.send("你@我啦～有什麽想和我説的嗎？", reference=message, mention_author=False)
            else:
                await send_message(message, user_message, channel_id, user_id, username, False)
        # 处理私信场景
        elif isinstance(message.channel, discord.DMChannel):
            await send_message(message, user_message, channel_id, user_id, username, True)

    client.run(token)
