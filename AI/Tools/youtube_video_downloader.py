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

"""
下载 YouTube 视频并返回视频信息字典

url: 视频链接

output_dir: 输出目录

format: 画质音频定义
- 最高画质音频 bv*+ba/b z
- 1080p bestvideo[height<=1080]+bestaudio/best[height<=1080]

merge_format: 合并格式 (mp4, mkv, etc.)

quiet: 是否静默模式

info 字典，下载失败返回 None
"""
def download_youtube_video(url, output_dir="youtube_video", format="bestvideo[height<=1080]+bestaudio/best[height<=1080]", merge_format="mp4", quiet=False):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        # 画质与音频
        "format": format,

        # 输出格式
        "merge_output_format": merge_format,

        # 文件名
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),

        # 不下载字幕
        "writesubtitles": False,
        "writeautomaticsub": False,

        # 显示下载进度
        "quiet": quiet,
        "no_warnings": False,

        # 避免文件名出现非法字符
        "restrictfilenames": False,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return info

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


# 视频 Url
video_url = "https://www.youtube.com/watch?v=sEwJvGi8KfE"

# 输出目录
output_dir = "youtube_video"

print(f"🚀 开始下载...\nURL: {video_url}\n")

# 执行下载
info = download_youtube_video(video_url, output_dir)

if info != None:
    print(f"\n✅ 下载完成")
    print(f"标题：{info.get('title')}")
    print(f"视频时长：{info.get('duration')} 秒")
