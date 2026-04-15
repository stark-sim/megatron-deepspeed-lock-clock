#!/usr/bin/env python3
"""
Generate PowerPoint presentation for GPU energy optimization project.
Output: GPU能效优化_20%+功耗节省方案.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Color scheme - warm beige theme
WARM_BEIGE = RGBColor(245, 240, 230)
DARK_BROWN = RGBColor(60, 40, 30)
ACCENT_GOLD = RGBColor(180, 140, 80)
SUCCESS_GREEN = RGBColor(80, 140, 80)
ALERT_RED = RGBColor(180, 80, 60)
LIGHT_GRAY = RGBColor(200, 200, 200)

def set_slide_bg(slide, color=WARM_BEIGE):
    """Set slide background color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_title_slide(prs, title, subtitle=""):
    """Add title slide."""
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    set_slide_bg(slide)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(14), Inches(1.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = DARK_BROWN
    p.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(1), Inches(4.2), Inches(14), Inches(1))
        tf = sub_box.text_frame
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(24)
        p.font.color.rgb = ACCENT_GOLD
        p.alignment = PP_ALIGN.CENTER
    
    return slide

def add_section_slide(prs, section_title):
    """Add section divider slide."""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    set_slide_bg(slide)
    
    # Section title
    title_box = slide.shapes.add_textbox(Inches(1), Inches(3), Inches(14), Inches(2))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = section_title
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = ACCENT_GOLD
    p.alignment = PP_ALIGN.CENTER
    
    return slide

def add_content_slide(prs, title, bullets, left_col=None, right_col=None):
    """Add content slide with title and bullets."""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    set_slide_bg(slide)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(15), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = DARK_BROWN
    
    # Main content area
    if left_col and right_col:
        # Two column layout
        # Left column
        left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(7), Inches(5.5))
        tf = left_box.text_frame
        tf.word_wrap = True
        
        for i, (header, items) in enumerate(left_col):
            if i > 0:
                p = tf.add_paragraph()
                p.text = ""
                p.space_before = Pt(12)
            
            p = tf.add_paragraph()
            p.text = header
            p.font.size = Pt(18)
            p.font.bold = True
            p.font.color.rgb = DARK_BROWN
            
            for item in items:
                p = tf.add_paragraph()
                p.text = "• " + item
                p.font.size = Pt(16)
                p.font.color.rgb = DARK_BROWN
                p.space_before = Pt(4)
                p.level = 0
        
        # Right column
        right_box = slide.shapes.add_textbox(Inches(8), Inches(1.3), Inches(7.5), Inches(5.5))
        tf = right_box.text_frame
        tf.word_wrap = True
        
        for i, (header, items) in enumerate(right_col):
            if i > 0:
                p = tf.add_paragraph()
                p.text = ""
                p.space_before = Pt(12)
            
            p = tf.add_paragraph()
            p.text = header
            p.font.size = Pt(18)
            p.font.bold = True
            p.font.color.rgb = DARK_BROWN
            
            for item in items:
                p = tf.add_paragraph()
                p.text = "• " + item
                p.font.size = Pt(16)
                p.font.color.rgb = DARK_BROWN
                p.space_before = Pt(4)
    else:
        # Single column bullets
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(15), Inches(5.5))
        tf = content_box.text_frame
        tf.word_wrap = True
        
        for bullet in bullets:
            p = tf.add_paragraph()
            p.text = "• " + bullet
            p.font.size = Pt(20)
            p.font.color.rgb = DARK_BROWN
            p.space_before = Pt(8)
    
    return slide

def add_highlight_slide(prs, title, highlight_text, sub_text=""):
    """Add slide with big highlight number/text."""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    set_slide_bg(slide)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(15), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = DARK_BROWN
    p.alignment = PP_ALIGN.CENTER
    
    # Highlight text
    highlight_box = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(14), Inches(1.5))
    tf = highlight_box.text_frame
    p = tf.paragraphs[0]
    p.text = highlight_text
    p.font.size = Pt(60)
    p.font.bold = True
    p.font.color.rgb = SUCCESS_GREEN
    p.alignment = PP_ALIGN.CENTER
    
    # Sub text
    if sub_text:
        sub_box = slide.shapes.add_textbox(Inches(1), Inches(4.2), Inches(14), Inches(1))
        tf = sub_box.text_frame
        p = tf.paragraphs[0]
        p.text = sub_text
        p.font.size = Pt(20)
        p.font.color.rgb = DARK_BROWN
        p.alignment = PP_ALIGN.CENTER
    
    return slide

def add_table_slide(prs, title, headers, rows):
    """Add slide with table."""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    set_slide_bg(slide)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(15), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = DARK_BROWN
    
    # Table
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table = slide.shapes.add_table(num_rows, num_cols, Inches(0.5), Inches(1.3), Inches(15), Inches(0.6 * num_rows)).table
    
    # Set column widths
    for i in range(num_cols):
        table.columns[i].width = Inches(15 / num_cols)
    
    # Header row
    for i, header in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = header
        cell.text_frame.paragraphs[0].font.size = Pt(14)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT_GOLD
        cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
    
    # Data rows
    for row_idx, row_data in enumerate(rows):
        for col_idx, cell_text in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = cell_text
            cell.text_frame.paragraphs[0].font.size = Pt(12)
            if "24%" in cell_text or "33%" in cell_text or "35%" in cell_text:
                cell.text_frame.paragraphs[0].font.color.rgb = SUCCESS_GREEN
                cell.text_frame.paragraphs[0].font.bold = True
    
    return slide

def create_presentation():
    """Create the complete presentation."""
    prs = Presentation()
    prs.slide_width = Inches(16)
    prs.slide_height = Inches(9)
    
    # Slide 1: Title
    add_title_slide(prs, 
        "分布式训练 GPU 能效优化",
        "基于锁频策略的 20%+ 功耗节省方案")
    
    # Slide 2: Outline
    slide = add_content_slide(prs, "汇报提纲", [])
    bullets = [
        "背景与挑战 - 大模型训练的能耗问题",
        "技术方案 - 三层优化体系与网络感知预测",
        "实验结果 - 24-35% 功耗节省实证",
        "应用价值 - 成本分析与落地场景",
        "总结与展望 - 核心成果与未来计划"
    ]
    tf = prs.slides[-1].shapes.add_textbox(Inches(2), Inches(2), Inches(12), Inches(5)).text_frame
    for bullet in bullets:
        p = tf.add_paragraph()
        p.text = bullet
        p.font.size = Pt(24)
        p.space_before = Pt(16)
        p.font.color.rgb = DARK_BROWN
    
    # Section: Background
    add_section_slide(prs, "背景与挑战")
    
    # Slide 3: Energy Challenges
    add_content_slide(prs, "大模型训练的能耗挑战", [],
        left_col=[
            ("行业痛点:", [
                "GPT-4 级别模型训练耗电可达 1,200 万度",
                "电费占总成本的 40-60%",
                "碳排放成为 AI 发展的社会责任约束"
            ]),
            ("传统方案局限:", [
                "动态频率调整复杂度高",
                "默认配置远非能效最优",
                "缺乏精确的能效预测工具"
            ])
        ],
        right_col=[
            ("我们的洞察:", [
                "固定频率策略可以在不显著影响训练时间的前提下",
                "实现显著的能耗节省"
            ])
        ]
    )
    
    # Slide 4: Technical Challenges
    add_content_slide(prs, "核心技术挑战", [
        "挑战 1: 频率-性能非线性 - 高频收益递减，存在饱和区",
        "挑战 2: 跨节点通信 - 网络环境影响预测准确性（高速网 vs VPN）",
        "挑战 3: 并行拓扑多样性 - TP/PP/DP 不同组合的挑战"
    ])
    
    # Section: Solution
    add_section_slide(prs, "技术方案")
    
    # Slide 5: System Architecture
    add_content_slide(prs, "系统架构: 三层优化体系", [
        "实验层: 标准化定频实验流程，支持多节点并行",
        "监控层: 毫秒级功率采集，精确到每个训练步",
        "预测层: 动态网络检测，自适应跨节点建模"
    ])
    
    # Slide 6: Key Innovation
    add_content_slide(prs, "关键创新: 网络感知预测", [],
        left_col=[
            ("传统方案:", [
                "固定跨节点惩罚系数",
                "假设慢速网络（VPN）",
                "在高速网络（IB）下失效",
                "预测误差高达 98.5%"
            ])
        ],
        right_col=[
            ("我们的方案:", [
                "实时测量网络带宽",
                "动态调整惩罚系数",
                "支持多种网络类型",
                "误差降至 11.48%"
            ])
        ]
    )
    
    # Section: Results
    add_section_slide(prs, "实验结果")
    
    # Slide 7: Core Metric
    add_highlight_slide(prs, 
        "核心指标: 功耗节省超过 20%",
        "平均功耗节省 24-35%",
        "实验配置: V100 GPU + Qwen2.5-7B + InfiniBand HDR")
    
    # Slide 8: Detailed Results
    add_table_slide(prs, "详细数据: 不同拓扑的能效对比",
        ["拓扑", "基线功率", "优化频点", "优化后功率", "节省"],
        [
            ["TP=1, PP=4, DP=4", "3170.6 W", "1252-1267 MHz", "2407.7 W", "24%"],
            ["TP=2, PP=2, DP=4", "3686.1 W", "1072-1125 MHz", "2477.3 W", "33%"],
            ["TP=4, PP=1, DP=4", "3613.0 W", "1080-1155 MHz", "2363.6 W", "35%"]
        ]
    )
    
    # Slide 9: Prediction Accuracy
    add_content_slide(prs, "预测准确性: 从 98.5% 到 11.48%", [],
        left_col=[
            ("修复前（固定系数）:", [
                "时间预测误差: 98.5%",
                "严重高估跨节点开销",
                "不适用于高速网络"
            ])
        ],
        right_col=[
            ("修复后（网络感知）:", [
                "时间预测误差: 11.48%",
                "准确识别网络特性",
                "支持多网络类型",
                "实现零样本优化"
            ])
        ]
    )
    
    # Section: Business Value
    add_section_slide(prs, "应用价值")
    
    # Slide 10: Cost Savings
    add_highlight_slide(prs,
        "成本节省估算",
        "单次训练节省 6,480 元",
        "以 100 卡集群训练 30 天为例，年化节省 77,760 元")
    
    # Slide 11: Deployment
    add_content_slide(prs, "落地场景与扩展性", [
        "适用场景: 企业 AI 训练集群、云 GPU 实例、超算中心、边缘节点",
        "部署要求: NVIDIA GPU (V100/A100/H100)、NVML 支持、DeepSpeed/PyTorch",
        "扩展路线: 网络层 (IB→RoCE→以太网)、拓扑层 (2×4→2×8→4×8)、硬件层 (V100→A100→H100)"
    ])
    
    # Section: Conclusion
    add_section_slide(prs, "总结与展望")
    
    # Slide 12: Core Achievements
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(15), Inches(0.8))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "核心成果总结"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = DARK_BROWN
    
    # Three boxes
    boxes = [
        ("20%+", "功耗节省", "所有测试拓扑均达标"),
        ("11.48%", "预测误差", "网络感知模型高精度"),
        ("1700×", "系数优化", "动态适配多网络类型")
    ]
    
    for i, (big, small, desc) in enumerate(boxes):
        x = 0.5 + i * 5.2
        # Box
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(2), Inches(4.5), Inches(2.5))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(240, 245, 240)
        shape.line.color.rgb = SUCCESS_GREEN
        
        # Big number
        box = slide.shapes.add_textbox(Inches(x), Inches(2.2), Inches(4.5), Inches(1))
        tf = box.text_frame
        p = tf.paragraphs[0]
        p.text = big
        p.font.size = Pt(36)
        p.font.bold = True
        p.font.color.rgb = SUCCESS_GREEN
        p.alignment = PP_ALIGN.CENTER
        
        # Small text
        box = slide.shapes.add_textbox(Inches(x), Inches(3.2), Inches(4.5), Inches(0.5))
        tf = box.text_frame
        p = tf.paragraphs[0]
        p.text = small
        p.font.size = Pt(20)
        p.font.color.rgb = DARK_BROWN
        p.alignment = PP_ALIGN.CENTER
        
        # Description
        box = slide.shapes.add_textbox(Inches(x), Inches(4), Inches(4.5), Inches(0.8))
        tf = box.text_frame
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(14)
        p.font.color.rgb = DARK_BROWN
        p.alignment = PP_ALIGN.CENTER
    
    # Bullet points below
    tf = slide.shapes.add_textbox(Inches(1), Inches(5.5), Inches(14), Inches(2)).text_frame
    bullets = [
        "首创动态网络感知跨节点性能预测框架",
        "预测精度飞跃：98.5% → 11.48%",
        "支持零样本频率优化，实验成本降低 50%+"
    ]
    for bullet in bullets:
        p = tf.add_paragraph()
        p.text = "• " + bullet
        p.font.size = Pt(18)
        p.space_before = Pt(8)
        p.font.color.rgb = DARK_BROWN
    
    # Slide 13: Next Steps
    add_content_slide(prs, "下一步工作计划", [
        "近期（1-2 个月）: RoCE 环境验证、论文投稿（MLSys/SC/IPDPS）、开源代码发布",
        "中期（3-6 个月）: A100/H100 适配、更大规模验证（32-64 卡）、自动化部署工具",
        "长期（6-12 个月）: 多租户场景优化、异构集群支持、云服务集成"
    ])
    
    # Slide 14: Thank You
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide)
    
    tf = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(14), Inches(2)).text_frame
    p = tf.paragraphs[0]
    p.text = "谢谢!"
    p.font.size = Pt(60)
    p.font.bold = True
    p.font.color.rgb = DARK_BROWN
    p.alignment = PP_ALIGN.CENTER
    
    p = tf.add_paragraph()
    p.text = "问答环节"
    p.font.size = Pt(32)
    p.space_before = Pt(20)
    p.font.color.rgb = ACCENT_GOLD
    p.alignment = PP_ALIGN.CENTER
    
    # Contact info
    tf = slide.shapes.add_textbox(Inches(1), Inches(5.5), Inches(14), Inches(2)).text_frame
    info = [
        "项目地址: github.com/[org]/Megatron-DeepSpeed",
        "联系邮箱: [email@example.com]",
        "技术文档: memory-bank/ & .context/paper/"
    ]
    for line in info:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(16)
        p.space_before = Pt(8)
        p.font.color.rgb = DARK_BROWN
        p.alignment = PP_ALIGN.CENTER
    
    # Save presentation
    output_file = "GPU能效优化_20%+功耗节省方案.pptx"
    prs.save(output_file)
    print(f"✓ Generated: {output_file}")
    print(f"  Total slides: {len(prs.slides)}")

if __name__ == "__main__":
    create_presentation()
