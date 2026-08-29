# coding=utf-8
import os
from yt_dlp import YoutubeDL

"""
下载 Youtube 视频，基于 yt_dlp

dependency packages
!pip -q install -U yt-dlp
!apt-get -qq update
!apt-get -qq install -y ffmpeg
"""
VIDEO_URL = "https://www.youtube.com/watch?v=sEwJvGi8KfE"

# 输出目录
OUTPUT_DIR = "youtube_video"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 下载配置
ydl_opts = {
    # 最高画质视频 + 最佳音频
    # 如果无法获取，则退回到最佳单文件格式
    # "format": "bv*+ba/b",

    # 1080p
    "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",

    # 自动合并为 MP4
    "merge_output_format": "mp4",

    # 文件名
    "outtmpl": os.path.join(OUTPUT_DIR, "%(title)s.%(ext)s"),

    # 不下载字幕
    "writesubtitles": False,
    "writeautomaticsub": False,

    # 显示下载进度
    "quiet": False,
    "no_warnings": False,

    # 避免文件名出现非法字符
    "restrictfilenames": False,
}

print("🚀 开始下载...")
print(f"URL: {VIDEO_URL}")
print()

with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(VIDEO_URL, download=True)
    print()
    print("✅ 下载完成")
    print(f"标题：{info.get('title')}")
