#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit 前端 - Apex Legends 缩圈预测练习

功能：
- 从离线数据集中随机抽取一场比赛（已知第 1、2 圈，已知真实第 5 圈）
- 让用户在「地图图像」上点击预测第 5 圈圈心位置
- 在地图上画出：第 1 圈（蓝色）、第 2 圈（青色）、用户预测（红色）、真实答案（绿色）
- 计算用户预测与真实圈心的距离，判断「正确 / 错误」

依赖：
- streamlit
- streamlit-image-coordinates  (pip install streamlit-image-coordinates)
- pandas, numpy, pillow

运行方式：
    streamlit run streamlit_app.py
"""

import os
import math
import random
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


# ===================== 基础配置 =====================

# 统一的地图坐标范围（训练数据中假定 0 ~ 20000）
MAP_MAX_COORD = 20000
# 用于前端展示的画布大小（像素）
CANVAS_SIZE = 800
MAP_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_image")

# 地图名称 -> 图片文件名映射（匹配你刚上传的文件名）
MAP_NAME_TO_FILE = {
    # 英文名
    "King's Canyon": "KingsCanyon.png",
    "World's Edge": "world's edge_副本.png",
    "Olympus": "Olympus_副本.png",
    "Broken Moon": "broken Moon_副本.png",
    "Storm Point": "storm point_副本.png",
    "E-District": "E-District_副本.png",
    # 中文名
    "诸王峡谷": "KingsCanyon.png",
    "世界边缘": "world's edge_副本.png",
    "奥林匹斯": "Olympus_副本.png",
    "残月": "broken Moon_副本.png",
    "风暴点": "storm point_副本.png",
    "电竞区": "E-District_副本.png",
}



@st.cache_data
def load_dataset(prefer_real: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    加载缩圈特征数据集（优先使用真实数据 output_real，其次 output）

    Returns:
        (df, source_dir)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if prefer_real:
        candidates.append(os.path.join(base_dir, "output_real"))
    candidates.append(os.path.join(base_dir, "output"))

    for dirname in candidates:
        csv_path = os.path.join(dirname, "algs_ring_dataset.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                required_cols = [
                    "ring1_x", "ring1_y",
                    "ring2_x", "ring2_y",
                    "target_ring5_x", "target_ring5_y",
                ]
                if not all(col in df.columns for col in required_cols):
                    continue
                df = df.dropna(subset=required_cols)
                if df.empty:
                    continue
                return df, dirname
            except Exception:
                pass

    return None, None


def game_to_pixel(x: float, y: float, w: int, h: int) -> Tuple[float, float]:
    """将游戏坐标 (0~MAP_MAX_COORD) 映射到图片像素坐标"""
    px = (x / MAP_MAX_COORD) * w
    py = (y / MAP_MAX_COORD) * h
    return px, py


def pixel_to_game(px: float, py: float, w: int, h: int) -> Tuple[float, float]:
    """将图片像素坐标映射回游戏坐标系"""
    gx = (px / max(w, 1)) * MAP_MAX_COORD
    gy = (py / max(h, 1)) * MAP_MAX_COORD
    return gx, gy


def sample_round(df: pd.DataFrame) -> pd.Series:
    """从数据集中随机抽取一行样本"""
    idx = random.randrange(len(df))
    return df.iloc[idx]


def find_map_image(map_cn: str, map_en: str) -> Optional[str]:
    """
    根据地图名称查找本地图片，先用硬编码映射，再做模糊匹配。
    """
    # 1. 优先使用硬编码映射
    for name in [map_en, map_cn]:
        if name and name in MAP_NAME_TO_FILE:
            fpath = os.path.join(MAP_IMAGE_DIR, MAP_NAME_TO_FILE[name])
            if os.path.exists(fpath):
                return fpath

    # 2. 模糊匹配
    if not os.path.isdir(MAP_IMAGE_DIR):
        return None

    def _norm(s: str) -> str:
        return "".join(ch.lower() for ch in s if (ch.isalnum() or "\u4e00" <= ch <= "\u9fff"))

    keys = [k for k in [_norm(map_cn or ""), _norm(map_en or "")] if k]
    if not keys:
        return None

    for fname in os.listdir(MAP_IMAGE_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        name_norm = _norm(fname)
        for key in keys:
            if key in name_norm:
                return os.path.join(MAP_IMAGE_DIR, fname)

    return None


def create_blank_map() -> Image.Image:
    """创建一个带网格的空白地图"""
    img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), color=(25, 27, 35))
    draw = ImageDraw.Draw(img)
    # 画网格
    grid_count = 10
    step = CANVAS_SIZE // grid_count
    for i in range(grid_count + 1):
        pos = i * step
        draw.line([(pos, 0), (pos, CANVAS_SIZE)], fill=(50, 55, 70), width=1)
        draw.line([(0, pos), (CANVAS_SIZE, pos)], fill=(50, 55, 70), width=1)
    return img


def draw_circles_on_map(
    base_img: Image.Image,
    sample: pd.Series,
    click_px: Optional[Tuple[float, float]] = None,
    show_answer: bool = False,
) -> Image.Image:
    """
    在地图上画圈：
    - 蓝色：第 1 圈
    - 青色：第 2 圈
    - 红色十字 + 圈：用户点击预测位置
    - 绿色十字 + 圈：真实第 5 圈答案（只有 show_answer=True 时）
    """
    img = base_img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # 核心修复：数据集里的 x,y 坐标是游戏 native 单位（~20000范围），而半径 r 通常是“米”
    # 比如 ring1_r = ~900 米。整个 Apex 地图宽度一般认为是 ~4000 米左右。
    MAP_WIDTH_METERS = 4000.0

    # --- 画第 1 圈（蓝色）---
    use_norm = "ring1_x_norm" in sample
    
    r1_r = float(sample.get("ring1_r", 900))
    rr1 = (r1_r / MAP_WIDTH_METERS) * w

    if use_norm:
        cx1 = float(sample["ring1_x_norm"]) * w
        cy1 = float(sample["ring1_y_norm"]) * h
    else:
        r1_x, r1_y = float(sample["ring1_x"]), float(sample["ring1_y"])
        cx1, cy1 = game_to_pixel(r1_x, r1_y, w, h)

    # ALGS 风格的大圈，白色/亮蓝色边框，内部有微微的透明白色/蓝色，表示安全区
    draw.ellipse(
        (cx1 - rr1, cy1 - rr1, cx1 + rr1, cy1 + rr1),
        fill=(255, 255, 255, 15),
        outline=(200, 230, 255, 255),
        width=5, 
    )
    # 用十字或者准星标记一下1圈中心，ALGS里有时候不需要，但我们这里保留隐约的标记
    draw.line([(cx1 - 10, cy1), (cx1 + 10, cy1)], fill=(200, 230, 255, 180), width=2)
    draw.line([(cx1, cy1 - 10), (cx1, cy1 + 10)], fill=(200, 230, 255, 180), width=2)

    # --- 画第 2 圈（青色）---
    if show_answer:
        r2_r = float(sample.get("ring2_r", 450))
        rr2 = (r2_r / MAP_WIDTH_METERS) * w

        if use_norm and "ring2_x_norm" in sample:
            cx2 = float(sample["ring2_x_norm"]) * w
            cy2 = float(sample["ring2_y_norm"]) * h
        else:
            r2_x, r2_y = float(sample["ring2_x"]), float(sample["ring2_y"])
            cx2, cy2 = game_to_pixel(r2_x, r2_y, w, h)
        
        # 显示缩小后的 2圈 范围，用略微不同的颜色
        draw.ellipse(
            (cx2 - rr2, cy2 - rr2, cx2 + rr2, cy2 + rr2),
            fill=(0, 230, 230, 20),
            outline=(0, 255, 255, 200),
            width=3,
        )

    # --- 用户预测（红色）---
    if click_px is not None:
        cx, cy = click_px
        # 十字线
        cross_sz = 12
        draw.line([(cx - cross_sz, cy), (cx + cross_sz, cy)], fill=(255, 60, 60), width=3)
        draw.line([(cx, cy - cross_sz), (cx, cy + cross_sz)], fill=(255, 60, 60), width=3)
        # 圆圈
        pred_r = w * 0.03
        draw.ellipse(
            (cx - pred_r, cy - pred_r, cx + pred_r, cy + pred_r),
            outline=(255, 60, 60, 230),
            width=3,
        )

    # --- 真实第 5 圈答案（绿色）---
    if show_answer:
        t5_x = float(sample["target_ring5_x"])
        t5_y = float(sample["target_ring5_y"])
        t5_r = float(sample.get("target_ring5_r", 50))
        tcx, tcy = game_to_pixel(t5_x, t5_y, w, h)
        tr5 = (t5_r / MAP_WIDTH_METERS) * w
        # 让半径可见，最少 8 像素
        tr5 = max(tr5, 8)
        draw.ellipse(
            (tcx - tr5, tcy - tr5, tcx + tr5, tcy + tr5),
            fill=(0, 255, 70, 40),
            outline=(0, 255, 70, 255),
            width=3,
        )
        # 十字线
        cross_sz = 14
        draw.line([(tcx - cross_sz, tcy), (tcx + cross_sz, tcy)], fill=(0, 255, 70), width=3)
        draw.line([(tcx, tcy - cross_sz), (tcx, tcy + cross_sz)], fill=(0, 255, 70), width=3)

        # 如果有点击预测，画一条连线从预测到真实
        if click_px is not None:
            cx, cy = click_px
            draw.line(
                [(cx, cy), (tcx, tcy)],
                fill=(255, 255, 100, 150),
                width=2,
            )

    return img


def main():
    st.set_page_config(
        page_title="Apex Ring Predictor",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ========== 自定义 CSS ==========
    st.markdown("""
    <style>
        /* 标题样式 */
        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }
        /* 指标数字 */
        [data-testid="stMetricValue"] {
            font-size: 1.4rem !important;
            font-weight: 700 !important;
        }
        /* 成功/错误消息 */
        .stAlert {
            border-radius: 8px !important;
        }
        /* 图例 */
        .legend-item {
            display: inline-flex;
            align-items: center;
            margin-right: 18px;
            font-size: 0.95rem;
            font-weight: 500;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
            border: 2px solid rgba(255,255,255,0.3);
            box-shadow: 0 0 4px rgba(0,0,0,0.5);
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 Apex Legends Ring Prediction Practice")

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            'Max error distance for "Correct" prediction (in-game units)',
            min_value=100,
            max_value=3000,
            value=800,
            step=50,
            help="Smaller values are stricter. 800 is approx 4% of total map width.",
        )

        st.markdown("---")

        if st.button("🎲 Draw New Match", use_container_width=True):
            st.session_state.pop("current_sample", None)
            st.session_state.pop("last_click", None)
            st.session_state.pop("submitted", None)
            st.rerun()

        if st.button("🗑️ Clear Prediction", use_container_width=True):
            st.session_state.pop("last_click", None)
            st.session_state.pop("submitted", None)
            st.rerun()

        st.markdown("---")
        st.markdown("""
        ### 📖 How to Play
        1. Observe the **Blue Ring** (Ring 1) on the map.
        2. **Click** on the map to predict the final **Ring 5** center.
        3. Click **Submit Prediction** to check your results.
        4. The system will reveal **Ring 2** and the **True Ring 5**.
        """)

        st.markdown("---")
        st.markdown("""
        ### 🏷️ Legend
        <div style="margin-top: 8px;">
            <div class="legend-item"><span class="legend-dot" style="background: #3c96ff;"></span> Ring 1</div>
            <div class="legend-item"><span class="legend-dot" style="background: #00e6e6;"></span> Ring 2</div>
            <div class="legend-item"><span class="legend-dot" style="background: #ff3c3c;"></span> Your Prediction</div>
            <div class="legend-item"><span class="legend-dot" style="background: #00ff46;"></span> True Ring 5</div>
        </div>
        """, unsafe_allow_html=True)

        # 统计
        if "stats" not in st.session_state:
            st.session_state.stats = {"total": 0, "correct": 0}
        stats = st.session_state.stats
        if stats["total"] > 0:
            st.markdown("---")
            st.markdown("### 📊 Statistics")
            accuracy = stats["correct"] / stats["total"] * 100
            st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
            st.progress(stats["correct"] / stats["total"])
            st.caption(f"{stats['correct']} Correct / {stats['total']} Total Matches")

    # ========== 加载数据 ==========
    df, source_dir = load_dataset(prefer_real=True)
    if df is None:
        st.error(
            "❌ Dataset not found.\n\n"
            "Please run in your terminal first:\n"
            "- `python workflow.py ./real_data ./output_real` to generate real dataset; or\n"
            "- `python train_model.py` to generate mock dataset."
        )
        return

    if "current_sample" not in st.session_state:
        st.session_state.current_sample = sample_round(df)

    sample = st.session_state.current_sample

    # 地图信息
    map_cn = sample.get("map_name_cn", "Unknown Map")
    map_en = sample.get("map_name_en", "")
    match_id = sample.get("match_id", "N/A")

    # ========== 主要两列布局 ==========
    col_info, col_map = st.columns([0.8, 2.2], gap="medium")

    with col_info:
        # 信息卡片
        st.subheader(f"🗺️ {map_en or map_cn}")
        st.caption(f"`{match_id}`")

        st.markdown("---")

        # 已知圈信息
        st.markdown("#### 📍 Known Ring Info")

        r1_x, r1_y = float(sample["ring1_x"]), float(sample["ring1_y"])
        r1_r = float(sample.get("ring1_r", 0))
        r2_x, r2_y = float(sample["ring2_x"]), float(sample["ring2_y"])
        r2_r = float(sample.get("ring2_r", 0))

        st.markdown(f"""
        | Ring | Radius (Units) | Status |
        |:---:|:---:|:---:|
        | 🔵 Ring 1 | {r1_r:.0f} | Given |
        | 🔷 Ring 2 | {r2_r:.0f} | Revealed after submit |
        """)

        # 提交按钮
        st.markdown("---")
        has_click = "last_click" in st.session_state and st.session_state.last_click is not None
        submitted = st.session_state.get("submitted", False)

        if has_click and not submitted:
            click_px = st.session_state.last_click
            pred_x, pred_y = pixel_to_game(click_px[0], click_px[1], CANVAS_SIZE, CANVAS_SIZE)
            st.info(f"🖱️ You clicked: ({pred_x:.0f}, {pred_y:.0f})")

            if st.button("✅ Submit Prediction", use_container_width=True, type="primary"):
                st.session_state.submitted = True
                # 更新统计
                true_x = float(sample["target_ring5_x"])
                true_y = float(sample["target_ring5_y"])
                distance = math.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
                st.session_state.stats["total"] += 1
                if distance <= threshold:
                    st.session_state.stats["correct"] += 1
                st.rerun()
        elif not has_click:
            st.warning("👆 Please click on the map to predict the Ring 5 center")

    with col_map:
        st.subheader("🗺️ Click on map to predict")

        # 加载地图图片
        map_path = find_map_image(map_cn, map_en)
        if map_path:
            base_img = Image.open(map_path).convert("RGB")
            base_img = base_img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
        else:
            base_img = create_blank_map()

        # 构建显示图像（附加圆圈）
        last_click = st.session_state.get("last_click")
        submitted = st.session_state.get("submitted", False)
        display_img = draw_circles_on_map(base_img, sample, last_click, show_answer=submitted)

        # streamlit-image-coordinates 组件
        # 使用一个唯一 key 防止重复触发
        click_result = streamlit_image_coordinates(
            display_img,
            key=f"map_click_{match_id}",
        )

        if click_result is not None:
            new_x = click_result["x"]
            new_y = click_result["y"]
            # 只有在还没提交的状态下，才更新点击位置
            if not submitted:
                current = st.session_state.get("last_click")
                if current is None or (current[0] != new_x or current[1] != new_y):
                    st.session_state.last_click = (new_x, new_y)
                    st.rerun()

    # ========== 结果反馈区域 ==========
    if submitted and has_click:
        st.markdown("---")

        click_px = st.session_state.last_click
        pred_x, pred_y = pixel_to_game(click_px[0], click_px[1], CANVAS_SIZE, CANVAS_SIZE)
        true_x = float(sample["target_ring5_x"])
        true_y = float(sample["target_ring5_y"])
        distance = math.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        is_correct = distance <= threshold

        # 三列展示结果
        st.subheader("📊 Prediction Results")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### 🔴 Your Prediction")
            st.metric(
                label="Predicted (x, y)",
                value=f"({pred_x:.0f}, {pred_y:.0f})",
            )

        with c2:
            st.markdown("##### 🟢 True Answer")
            st.metric(
                label="True (x, y)",
                value=f"({true_x:.0f}, {true_y:.0f})",
            )

        with c3:
            st.markdown("##### 📏 Error Evaluation")
            delta_pct = distance / MAP_MAX_COORD * 100
            st.metric(
                label="Euclidean Distance",
                value=f"{distance:.0f}",
                delta=f"{delta_pct:.1f}% Map Width",
                delta_color="off",
            )
            st.metric(label="Threshold", value=f"{threshold}")

        st.markdown("---")
        if is_correct:
            st.success(f"✅ Congrats! Error {distance:.0f} ({delta_pct:.1f}%) is within threshold {threshold}. Evaluated as **CORRECT**!")
        else:
            st.error(f"❌ Unfortunately, Error {distance:.0f} ({delta_pct:.1f}%) exceeds threshold {threshold}. Evaluated as **WRONG**.")

        # 下一局快捷按钮
        st.markdown("")
        if st.button("⏩ Next Match", use_container_width=True, type="primary"):
            st.session_state.pop("current_sample", None)
            st.session_state.pop("last_click", None)
            st.session_state.pop("submitted", None)
            st.rerun()

    # 底部信息
    st.markdown("---")
    st.caption(f"Data source: `{source_dir}` · Total {len(df)} matches")


if __name__ == "__main__":
    main()
