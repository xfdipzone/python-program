# coding=utf-8
import os
import glob
import html
import ipywidgets as widgets

from IPython.display import display, clear_output, HTML
from yt_dlp import YoutubeDL
from google.colab import files

"""
可视化下载 Youtube 视频，基于 yt_dlp

dependency packages
!pip -q install -U yt-dlp ipywidgets
!apt-get -qq update
!apt-get -qq install -y ffmpeg
"""
# 基础设置
OUTPUT_DIR = "youtube_video"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局变量
downloaded_file = None
video_info = None

# 格式化视频时长
def format_duration(seconds):
    if not seconds:
        return "未知"

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return f"{minutes}:{secs:02d}"

# 格式化文件大小
def format_size(size):
    if not size:
        return "未知"

    size = float(size)

    if size < 1024:
        return f"{size:.0f} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"

    if size < 1024 ** 3:
        return f"{size / 1024 ** 2:.1f} MB"

    return f"{size / 1024 ** 3:.2f} GB"


# UI
title = widgets.HTML(
    """
    <div style="font-size:28px; font-weight:bold; margin-bottom:5px;">
        🎬 YouTube Downloader
    </div>

    <div style="color:#666; margin-bottom:20px;">
        下载 YouTube 视频并保存为 MP4
    </div>
    """
)

# URL
url_input = widgets.Text(
    value="",
    placeholder="粘贴 YouTube 视频 URL",
    description="🔗 URL:",
    layout=widgets.Layout(
        width="680px"
    )
)

# 获取信息
info_button = widgets.Button(
    description="🔍 获取信息",
    button_style="",
    layout=widgets.Layout(
        width="130px",
        height="35px"
    )
)

# 画质
quality_dropdown = widgets.Dropdown(
    options=[
        ("1080p", "1080"),
        ("720p", "720"),
        ("480p", "480"),
        ("360p", "360"),
        ("最高画质", "best"),
    ],
    value="1080",
    description="🎞 画质:",
    layout=widgets.Layout(
        width="280px"
    )
)

# 下载按钮
download_button = widgets.Button(
    description="⬇ 开始下载",
    button_style="primary",
    layout=widgets.Layout(
        width="140px",
        height="35px"
    )
)

# 保存按钮
save_button = widgets.Button(
    description="💾 保存到电脑",
    button_style="success",
    disabled=True,
    layout=widgets.Layout(
        width="150px",
        height="35px"
    )
)

# 状态
status = widgets.HTML(
    """
    <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#f5f5f5; border-radius:6px;">
        <b>状态：</b>等待操作
    </div>
    """
)

# 下载进度
progress = widgets.FloatProgress(
    value=0,
    min=0,
    max=100,
    step=0.1,
    description="进度:",
    bar_style="",
    layout=widgets.Layout(
        width="700px"
    )
)

# 下载详细信息
download_status = widgets.HTML(
    """
    <div style="width:680px; margin-top:5px; color:#666;">
        等待下载...
    </div>
    """
)

# 视频信息输出
info_output = widgets.Output()

# 下载完成信息
result_output = widgets.Output()

# 获取视频信息
def get_video_info(button):
    global video_info
    url = url_input.value.strip()

    if not url:
        with info_output:
            clear_output()
            print("❌ 请先输入 YouTube URL")
        return

    # UI 状态
    info_button.disabled = True
    download_button.disabled = True
    save_button.disabled = True

    status.value = """
    <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#f5f5f5; border-radius:6px;">
        <b>状态：</b>🔍 正在获取视频信息...
    </div>
    """

    with info_output:
        clear_output()

    try:
        # 获取视频信息
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(
                url,
                download=False
            )

        # 信息
        title_text = html.escape(
            video_info.get(
                "title",
                "未知"
            )
        )

        # 作者
        uploader = html.escape(
            video_info.get(
                "uploader",
                "未知"
            )
        )

        # 视频长度（秒）
        duration = format_duration(
            video_info.get(
                "duration"
            )
        )

        # 视频缩略图
        thumbnail = video_info.get(
            "thumbnail"
        )

        # 显示信息
        with info_output:
            display(
                HTML(
                    f"""
                    <div style="width:680px; border:1px solid #ddd; border-radius:10px; padding:15px; margin-top:5px;">

                        <div style="display:flex; gap:20px;">

                            <img
                                src="{thumbnail}"
                                width="220"
                                height="124"
                                style="object-fit:cover; border-radius:8px;"
                            >

                            <div>

                                <div style="font-size:18px; font-weight:bold; margin-bottom:12px;">
                                    {title_text}
                                </div>

                                <div style="margin-bottom:8px;">
                                    👤 作者：
                                    {uploader}
                                </div>

                                <div>
                                    ⏱ 时长：
                                    {duration}
                                </div>

                            </div>

                        </div>

                    </div>
                    """
                )
            )

        status.value = """
        <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#f5f5f5; border-radius:6px;">
            <b>状态：</b>
            <span style="color:green;">
                ✅ 视频信息获取成功
            </span>
        </div>
        """

        download_button.disabled = False

    except Exception as e:

        with info_output:
            clear_output()
            print("❌ 获取视频信息失败")
            print()
            print(str(e))

        status.value = """
        <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#fff3f3; border-radius:6px;">
            <b>状态：</b>
            <span style="color:red;">
                ❌ 获取视频信息失败
            </span>
        </div>
        """

    finally:
        info_button.disabled = False

# 下载进度 Hook
def progress_hook(data):
    # 下载中
    if data["status"] == "downloading":

        total = (
            data.get("total_bytes")
            or data.get("total_bytes_estimate")
        )

        downloaded = data.get(
            "downloaded_bytes",
            0
        )

        # 百分比
        if total:
            percent = (
                downloaded / total * 100
            )

            progress.value = min(
                percent,
                100
            )

        # 速度
        speed = data.get(
            "_speed_str",
            "未知"
        )

        # ETA
        eta = data.get(
            "_eta_str",
            "未知"
        )

        # 当前文件
        filename = data.get(
            "filename",
            ""
        )

        filename = os.path.basename(
            filename
        )

        filename = html.escape(
            filename
        )

        # UI
        download_status.value = f"""
        <div style="width:680px; margin-top:5px; line-height:1.7;">

            <div>
                📄 {filename}
            </div>

            <div>
                📊 <b>{progress.value:.1f}%</b>
                &nbsp;&nbsp;
                ⚡ {speed}
                &nbsp;&nbsp;
                ⏱ {eta}
            </div>

        </div>
        """

        status.value = f"""
        <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#f5f5f5; border-radius:6px;">
            <b>状态：</b>
            ⬇ 正在下载
            <b>{progress.value:.1f}%</b>
        </div>
        """

    # 单个文件下载完成
    elif data["status"] == "finished":
        progress.value = 100

        download_status.value = """
        <div style="width:680px; margin-top:5px; line-height:1.7;">
            📦 视频/音频下载完成<br>
            🔄 正在合并为 MP4...
        </div>
        """

        status.value = """
        <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#f5f5f5; border-radius:6px;">
            <b>状态：</b>
            🔄 正在合并视频和音频...
        </div>
        """

# 开始下载
def download_video(button):
    global downloaded_file
    url = url_input.value.strip()

    if not url:
        status.value = """
        <div style="padding:10px; background:#fff3f3; border-radius:6px;">
            <b>状态：</b>
            <span style="color:red;">
                ❌ 请先输入 YouTube URL
            </span>
        </div>
        """

        return

    # UI 初始化
    progress.value = 0
    download_button.disabled = True
    info_button.disabled = True
    save_button.disabled = True
    downloaded_file = None

    download_status.value = """
    <div style="width:680px; margin-top:5px;">
        准备下载...
    </div>
    """

    with result_output:
        clear_output()

    # 清理之前下载的文件
    for f in glob.glob(
        os.path.join(
            OUTPUT_DIR,
            "*"
        )
    ):

        try:
            if os.path.isfile(f):
                os.remove(f)

        except:
            pass

    # 画质
    quality = quality_dropdown.value

    if quality == "best":
        format_option = (
            "bestvideo+bestaudio/best"
        )

    else:
        format_option = (
            f"bestvideo[height<={quality}]"
            f"+bestaudio/"
            f"best[height<={quality}]"
        )

    # yt-dlp 配置
    ydl_opts = {
        # 视频 + 音频
        "format": format_option,

        # 合并 MP4
        "merge_output_format": "mp4",

        # 文件名
        "outtmpl": os.path.join(
            OUTPUT_DIR,
            "%(title)s.%(ext)s"
        ),

        # 进度 Hook
        "progress_hooks": [
            progress_hook
        ],

        # 不显示 yt-dlp 原始日志
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,

        # 不下载字幕
        "writesubtitles": False,
        "writeautomaticsub": False,

        # 不下载缩略图
        "writethumbnail": False,

        # 保留原始文件名
        "restrictfilenames": False,
    }

    try:
        # 开始下载
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                url,
                download=True
            )

        # 查找 MP4
        mp4_files = glob.glob(
            os.path.join(
                OUTPUT_DIR,
                "*.mp4"
            )
        )

        if not mp4_files:
            raise Exception(
                "下载完成，但没有找到 MP4 文件"
            )

        downloaded_file = max(
            mp4_files,
            key=os.path.getmtime
        )

        # 文件大小
        file_size = os.path.getsize(
            downloaded_file
        )

        # 完成
        progress.value = 100

        download_status.value = f"""
        <div style="width:680px; margin-top:5px; line-height:1.8;">

            <div>
                📄 文件：
                <b>
                    {html.escape(
            os.path.basename(
                downloaded_file
            )
        )}
                </b>
            </div>

            <div>
                💾 大小：
                <b>{format_size(file_size)}</b>
            </div>

            <div>
                🎞 画质：
                <b>{quality}</b>
            </div>

        </div>
        """

        status.value = """
        <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#f0fff4; border-radius:6px;">
            <b>状态：</b>
            <span style="color:green;">
                ✅ 下载完成
            </span>
        </div>
        """

        # 显示结果
        with result_output:
            clear_output()

            display(
                HTML(
                    """
                    <div style="margin-top:10px; margin-bottom:10px; color:green; font-weight:bold;">
                        🎉 视频已经准备完成
                    </div>
                    """
                )
            )

            display(save_button)

        save_button.disabled = False

    except Exception as e:

        status.value = """
        <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#fff3f3; border-radius:6px;">
            <b>状态：</b>
            <span style="color:red;">
                ❌ 下载失败
            </span>
        </div>
        """

        download_status.value = """
        <div style="width:680px; color:red;">
            下载失败，请查看下面的错误信息。
        </div>
        """

        with result_output:
            clear_output()

            print("❌ 下载失败")
            print()
            print(str(e))

    finally:
        download_button.disabled = False
        info_button.disabled = False


# 保存到电脑
def save_to_computer(button):
    global downloaded_file

    if not downloaded_file:
        return

    if not os.path.exists(
        downloaded_file
    ):

        with result_output:
            print("❌ 找不到下载文件")

        return

    status.value = """
    <div style="margin-top:10px; margin-bottom:10px; padding:10px; background:#f5f5f5; border-radius:6px;">
        <b>状态：</b>
        ⬇ 正在保存到电脑...
    </div>
    """

    files.download(
        downloaded_file
    )


# 按钮事件
info_button.on_click(
    get_video_info
)

download_button.on_click(
    download_video
)

save_button.on_click(
    save_to_computer
)

# 显示 UI
display(title)

# URL
display(
    widgets.HBox([
        url_input,
        info_button
    ])
)

# 画质 + 下载
display(
    widgets.HBox([
        quality_dropdown,
        download_button
    ])
)

# 状态
display(status)

# 进度条
display(progress)

# 下载状态
display(download_status)

# 视频信息
display(
    HTML(
        """
        <h4 style="margin-top:25px;">
            📺 视频信息
        </h4>
        """
    )
)

display(info_output)

# 下载结果
display(
    HTML(
        """
        <h4 style="margin-top:25px;">
            📥 下载状态
        </h4>
        """
    )
)

display(result_output)
