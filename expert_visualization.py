#!/usr/bin/env python3
"""
专家可视化分析脚本
生成专家学习的各种可视化图表
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple
import re


def load_router_data(dump_path: str) -> List[Dict]:
    """加载路由数据"""
    data = []
    with open(dump_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_expert_specialization(data: List[Dict], output_dir: str):
    """分析专家特化模式"""
    print("Analyzing expert specialization patterns...")

    # 1. 专家正确率分析
    expert_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in data:
        if "detailed_router_info" in record and record["detailed_router_info"]["token_routers"]:
            for router_name, router_info in record["detailed_router_info"]["token_routers"].items():
                if "top_experts" in router_info:
                    correct = record["correct_at_1"]
                    for expert in router_info["top_experts"]:
                        expert_accuracy[expert]["total"] += 1
                        if correct:
                            expert_accuracy[expert]["correct"] += 1

    # 2. 创建专家准确率图表
    experts = []
    accuracies = []
    for expert, stats in expert_accuracy.items():
        if stats["total"] > 0:
            experts.append(expert)
            accuracies.append(stats["correct"] / stats["total"] * 100)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(experts, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(experts))))
    plt.title("Expert Specialization Accuracy", fontsize=16, fontweight='bold')
    plt.xlabel("Expert ID")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 专家使用频率分析
    expert_usage = Counter()
    for record in data:
        if "detailed_router_info" in record:
            for router_name, router_info in record["detailed_router_info"]["token_routers"].items():
                if "top_experts" in router_info:
                    for expert in router_info["top_experts"]:
                        expert_usage[expert] += 1

    # 创建饼图
    plt.figure(figsize=(8, 8))
    plt.pie(expert_usage.values(), labels=expert_usage.keys(), autopct='%1.1f%%',
            startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(expert_usage))))
    plt.title("Expert Usage Distribution", fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_usage_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_question_expert_correlation(data: List[Dict], output_dir: str):
    """分析问题类型与专家选择的关联"""
    print("Analyzing question-expert correlation...")

    # 1. 问题分类函数
    def classify_question(question):
        question_lower = question.lower()

        # 地理区域
        if "nyc" in question_lower or "new york" in question_lower:
            region = "NYC"
        elif "tokyo" in question_lower or "tky" in question_lower:
            region = "Tokyo"
        elif "california" in question_lower or "ca" in question_lower:
            region = "California"
        else:
            region = "Unknown"

        # 查询类型
        if any(keyword in question_lower for keyword in ["near", "closest", "around"]):
            query_type = "Proximity"
        elif any(keyword in question_lower for keyword in ["route", "path"]):
            query_type = "Routing"
        elif any(keyword in question_lower for keyword in ["best", "top", "recommend"]):
            query_type = "Ranking"
        else:
            query_type = "Direct"

        return region, query_type

    # 2. 收集问题-专家对
    question_expert_pairs = defaultdict(lambda: defaultdict(int))

    for record in data:
        if "detailed_router_info" in record:
            question = record.get("question", "")
            region, query_type = classify_question(question)

            for router_name, router_info in record["detailed_router_info"]["token_routers"].items():
                if "top_experts" in router_info:
                    for expert in router_info["top_experts"]:
                        question_expert_pairs[query_type][expert] += 1

    # 3. 创建关联热力图
    query_types = list(question_expert_pairs.keys())
    experts = set()
    for q_type in query_types:
        experts.update(question_expert_pairs[q_type].keys())
    experts = sorted(experts)

    # 创建矩阵
    matrix = np.zeros((len(query_types), len(experts)))
    for i, q_type in enumerate(query_types):
        for j, expert in enumerate(experts):
            matrix[i, j] = question_expert_pairs[q_type][expert]

    # 绘制热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(matrix,
                xticklabels=[f"E{e}" for e in experts],
                yticklabels=query_types,
                annot=True,
                fmt='d',
                cmap="Blues")
    plt.title("Question Type - Expert Selection Correlation", fontsize=16, fontweight='bold')
    plt.xlabel("Expert ID")
    plt.ylabel("Question Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question_expert_correlation.png"), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_expert_cooperation(data: List[Dict], output_dir: str):
    """分析专家合作模式"""
    print("Analyzing expert cooperation patterns...")

    # 1. 专家共现矩阵
    expert_cooccurrence = defaultdict(lambda: defaultdict(int))

    for record in data:
        if "detailed_router_info" in record:
            all_experts = set()
            for router_name, router_info in record["detailed_router_info"]["token_routers"].items():
                if "top_experts" in router_info:
                    all_experts.update(router_info["top_experts"])

            # 统计专家共现
            expert_list = sorted(all_experts)
            for i, expert1 in enumerate(expert_list):
                for expert2 in expert_list[i+1:]:
                    expert_cooccurrence[expert1][expert2] += 1
                    expert_cooccurrence[expert2][expert1] += 1

    # 2. 创建共现网络图
    G = nx.Graph()

    for expert1 in expert_cooccurrence:
        for expert2 in expert_cooccurrence[expert1]:
            weight = expert_cooccurrence[expert1][expert2]
            G.add_edge(f"E{expert1}", f"E{expert2}", weight=weight)

    # 3. 绘制网络图
    plt.figure(figsize=(10, 8))

    # 计算节点位置
    pos = nx.spring_layout(G, k=1, iterations=50)

    # 获取边权重
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]

    # 绘制网络
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    plt.title("Expert Cooperation Network", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_cooperation_network.png"), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_expert_weight_distribution(data: List[Dict], output_dir: str):
    """分析专家权重分布"""
    print("Analyzing expert weight distributions...")

    # 收集所有权重
    all_weights = []
    weight_by_layer = defaultdict(list)

    for record in data:
        if "detailed_router_info" in record:
            for router_name, router_info in record["detailed_router_info"]["token_routers"].items():
                if "softmax_weights" in router_info and router_info["softmax_weights"]:
                    weights = router_info["softmax_weights"]
                    weight_by_layer[router_name].extend(weights)
                    all_weights.extend(weights)

    # 创建权重分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # 总体权重分布
    axes[0].hist(all_weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title("Overall Weight Distribution", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Weight Value")
    axes[0].set_ylabel("Frequency")

    # 各层的权重分布
    for i, (layer, weights) in enumerate(weight_by_layer.items()):
        if i >= 3:  # 最多显示3个层
            break

        axes[i+1].hist(weights, bins=30, alpha=0.7, label=layer)
        axes[i+1].set_title(f"Weight Distribution - {layer}", fontsize=12)
        axes[i+1].set_xlabel("Weight Value")
        axes[i+1].set_ylabel("Frequency")
        axes[i+1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_weight_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_expert_report(data: List[Dict], output_dir: str):
    """生成专家分析报告"""
    print("Generating expert analysis report...")

    # 统计基本信息
    total_samples = len(data)
    experts_used = set()

    for record in data:
        if "detailed_router_info" in record:
            for router_name, router_info in record["detailed_router_info"]["token_routers"].items():
                if "top_experts" in router_info:
                    experts_used.update(router_info["top_experts"])

    report = {
        "summary": {
            "total_samples": total_samples,
            "num_experts": len(experts_used),
            "experts_used": sorted(experts_used)
        },
        "findings": []
    }

    # 分析发现
    if len(experts_used) > 0:
        report["findings"].append(f"模型使用了 {len(experts_used)} 个专家")
        report["findings"].append(f"这些专家共同处理了 {total_samples} 个测试样本")
        report["findings"].append("专家之间存在不同的特化模式和合作方式")

    # 保存报告
    with open(os.path.join(output_dir, "expert_analysis_report.md"), 'w', encoding='utf-8') as f:
        f.write("# 专家学习内容分析报告\n\n")
        f.write(f"## 概述\n\n")
        f.write(f"- 总样本数: {total_samples}\n")
        f.write(f"- 使用的专家数: {len(experts_used)}\n")
        f.write(f"- 专家ID: {sorted(experts_used)}\n\n")

        f.write("## 分析发现\n\n")
        for finding in report["findings"]:
            f.write(f"- {finding}\n")

        f.write("\n## 可视化图表\n\n")
        f.write("1. 专家准确率分析\n")
        f.write("2. 问题类型与专家关联\n")
        f.write("3. 专家合作网络\n")
        f.write("4. 权重分布分析\n")

    print(f"Expert analysis report saved to: {output_dir}/expert_analysis_report.md")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Expert Visualization Analysis')
    parser.add_argument('--router_dump_path', type=str, required=True,
                        help='Path to the router dump JSONL file')
    parser.add_argument('--output_dir', type=str, default='./expert_visualization',
                        help='Output directory for visualization results')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    data = load_router_data(args.router_dump_path)

    # 运行各种分析
    analyze_expert_specialization(data, args.output_dir)
    analyze_question_expert_correlation(data, args.output_dir)
    analyze_expert_cooperation(data, args.output_dir)
    analyze_expert_weight_distribution(data, args.output_dir)
    generate_expert_report(data, args.output_dir)

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()