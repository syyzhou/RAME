#!/usr/bin/env python3
"""
专家学习内容分析脚本
分析专家的选择模式、权重分布和学习到的偏好
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from typing import Dict, List, Tuple
import re


def classify_question(question: str) -> Dict[str, str]:
    """对问题进行分类"""
    question_lower = question.lower()

    # 地理区域分类
    if "nyc" in question_lower or "new york" in question_lower:
        region = "NYC"
    elif "tokyo" in question_lower or "tky" in question_lower:
        region = "Tokyo"
    elif "california" in question_lower or "ca" in question_lower:
        region = "California"
    else:
        region = "Unknown"

    # 问题类型分类
    if any(keyword in question_lower for keyword in ["find", "locate", "where", "search"]):
        if any(keyword in question_lower for keyword in ["near", "closest", "around"]):
            query_type = "Proximity"
        else:
            query_type = "Direct"
    elif any(keyword in question_lower for keyword in ["route", "path", "how to get", "direction"]):
        query_type = "Routing"
    elif any(keyword in question_lower for keyword in ["best", "top", "recommend", "popular"]):
        query_type = "Ranking"
    else:
        query_type = "Other"

    # POI类型分类
    if any(keyword in question_lower for keyword in ["restaurant", "food", "eat", "dinner"]):
        poi_type = "Restaurant"
    elif any(keyword in question_lower for keyword in ["hotel", "accommodation", "stay"]):
        poi_type = "Hotel"
    elif any(keyword in question_lower for keyword in ["attraction", "tourist", "sightseeing", "museum"]):
        poi_type = "Attraction"
    elif any(keyword in question_lower for keyword in ["shopping", "mall", "store"]):
        poi_type = "Shopping"
    else:
        poi_type = "Other"

    # 复杂度分类
    if len(re.findall(r'\b(?:and|or|but|near|except|excluding)\b', question_lower)) > 1:
        complexity = "Complex"
    else:
        complexity = "Simple"

    return {
        "region": region,
        "query_type": query_type,
        "poi_type": poi_type,
        "complexity": complexity
    }


def analyze_expert_selections(router_dump_path: str, output_dir: str):
    """分析专家选择模式"""
    print(f"Loading router data from: {router_dump_path}")

    # 1. 收集所有数据
    all_data = []
    expert_selections = defaultdict(list)
    question_types = defaultdict(list)

    with open(router_dump_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data['question']

            # 分类问题
            question_info = classify_question(question)
            question_types["region"].append(question_info["region"])
            question_types["query_type"].append(question_info["query_type"])
            question_types["poi_type"].append(question_info["poi_type"])
            question_types["complexity"].append(question_info["complexity"])

            # 记录专家选择
            if "router_snapshot" in data:
                snapshot = data["router_snapshot"]

                # Token Router selections (专家选择)
                if "token_routers" in snapshot:
                    for router_name, router_info in snapshot["token_routers"].items():
                        if "top_experts" in router_info:
                            experts = router_info["top_experts"]
                            expert_selections[router_name].append({
                                "sample_id": data["sample_id"],
                                "question": question,
                                "experts": experts,
                                "weights": router_info.get("softmax_weights", []),
                                "question_info": question_info
                            })

                # Gate Router selections (门选择)
                if "gate_routers" in snapshot:
                    for gate_name, gate_info in snapshot["gate_routers"].items():
                        expert_selections[gate_name].append({
                            "sample_id": data["sample_id"],
                            "question": question,
                            "selected_group": gate_info.get("selected_group", []),
                            "weights": gate_info.get("group_weights", []),
                            "question_info": question_info
                        })

    # 2. 创建分析可视化目录
    viz_dir = os.path.join(output_dir, "expert_analysis_visualization")
    os.makedirs(viz_dir, exist_ok=True)

    # 3. 专家选择频率分析
    create_expert_selection_heatmap(expert_selections, viz_dir)

    # 4. 问题类型-专家偏好分析
    create_question_type_analysis(expert_selections, question_types, viz_dir)

    # 5. 专家权重分布分析
    create_weight_distribution_analysis(expert_selections, viz_dir)

    # 6. 专家模式聚类分析
    create_expert_clustering_analysis(expert_selections, viz_dir)

    # 7. 保存分析结果
    save_analysis_results(expert_selections, question_types, viz_dir)

    print(f"Expert analysis completed! Results saved to: {viz_dir}")


def create_expert_selection_heatmap(expert_selections: Dict, output_dir: str):
    """创建专家选择频率热力图"""
    print("Creating expert selection heatmap...")

    # 合并所有router的数据
    all_selections = {}
    for router_name, selections in expert_selections.items():
        if "top_experts" in selections[0]["experts"] if selections else False:
            # Token router
            layer_id = router_name.split("_")[0] if "_" in router_name else "unknown"
            expert_counts = Counter()

            for selection in selections:
                for expert in selection["experts"]:
                    expert_counts[f"{layer_id}_E{expert}"] += 1

            all_selections.update(expert_counts)

    if not all_selections:
        print("No expert selections found for heatmap")
        return

    # 创建矩阵
    layers = sorted(set([key.split("_")[0] for key in all_selections.keys()]))
    max_experts = max([max([int(key.split("_")[1].replace("E", "")) for key in all_selections.keys()])] + [0])

    # 初始化矩阵
    matrix = np.zeros((len(layers), max_experts))

    # 填充矩阵
    for key, count in all_selections.items():
        layer = key.split("_")[0]
        expert_num = int(key.split("_")[1].replace("E", ""))

        if layer in layers:
            layer_idx = layers.index(layer)
            if expert_num < max_experts:
                matrix[layer_idx, expert_num] = count

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix,
                xticklabels=[f"E{i}" for i in range(max_experts)],
                yticklabels=[f"L{i}" for i in layers],
                annot=True,
                fmt='d',
                cmap="YlOrRd")
    plt.title("Expert Selection Frequency Heatmap", fontsize=16, fontweight='bold')
    plt.xlabel("Expert ID", fontsize=12)
    plt.ylabel("Layer ID", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "expert_selection_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: expert_selection_heatmap.png")


def create_question_type_analysis(expert_selections: Dict, question_types: Dict, output_dir: str):
    """分析问题类型与专家选择的关联"""
    print("Creating question type analysis...")

    # 为每个问题类型创建专家偏好图表
    for type_category, type_values in question_types.items():
        if len(type_values) == 0:
            continue

        # 统计每个类型下的专家选择
        type_expert_prefs = defaultdict(lambda: defaultdict(int))

        for router_name, selections in expert_selections.items():
            if selections and "top_experts" in selections[0]:
                for i, selection in enumerate(selections):
                    question_type = type_values[i]

                    for expert in selection["experts"]:
                        type_expert_prefs[question_type][f"E{expert}"] += 1

        # 创建雷达图
        if len(type_expert_prefs) > 1:
            create_radar_chart(type_expert_prefs,
                             os.path.join(output_dir, f"expert_preference_by_{type_category}.png"),
                             f"Expert Preference by {type_category.title()}")


def create_weight_distribution_analysis(expert_selections: Dict, output_dir: str):
    """分析权重分布"""
    print("Creating weight distribution analysis...")

    # 收集所有权重
    all_weights = []
    weight_distributions = defaultdict(list)

    for router_name, selections in expert_selections.items():
        for selection in selections:
            if "weights" in selection and selection["weights"]:
                weights = selection["weights"]

                # 如果是列表形式，取前几个专家的权重
                if isinstance(weights, list):
                    weight_distributions[router_name].extend(weights[:len(weights)])
                elif isinstance(weights, dict):
                    # 如果是字典形式，取所有值
                    weight_distributions[router_name].extend(list(weights.values()))

    # 创建权重分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (router_name, weights) in enumerate(weight_distributions.items()):
        if i >= 4:  # 最多显示4个
            break

        if weights:
            axes[i].hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].set_title(f"{router_name}\nWeight Distribution", fontsize=12)
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weight_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: weight_distributions.png")


def create_expert_clustering_analysis(expert_selections: Dict, output_dir: str):
    """专家模式聚类分析"""
    print("Creating expert clustering analysis...")

    # 准备数据：每个样本的专家选择模式
    sample_patterns = []
    sample_ids = []

    for router_name, selections in expert_selections.items():
        if selections and "top_experts" in selections[0]:
            for selection in selections:
                # 创建专家选择模式向量
                pattern = [1 if i in selection["experts"] else 0 for i in range(4)]  # 假设4个专家
                sample_patterns.append(pattern)
                sample_ids.append(selection["sample_id"])

    if len(sample_patterns) > 10:  # 只有足够的样本才进行聚类
        patterns = np.array(sample_patterns)

        # 使用PCA降维
        pca = PCA(n_components=2)
        patterns_pca = pca.fit_transform(patterns)

        # 绘制聚类结果
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(patterns_pca[:, 0], patterns_pca[:, 1], alpha=0.6, c=range(len(patterns)))
        plt.colorbar(scatter, label="Sample Index")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title("Expert Selection Pattern Clustering", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "expert_pattern_clustering.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print("Saved: expert_pattern_clustering.png")


def create_radar_chart(data: Dict, output_path: str, title: str):
    """创建雷达图"""
    categories = list(data.keys())
    experts = set()
    for category_experts in data.values():
        experts.update(category_experts.keys())
    experts = list(experts)

    # 计算每个类别的归一化值
    angles = np.linspace(0, 2 * np.pi, len(experts), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for category in categories:
        values = [data[category][expert] for expert in experts]
        values += values[:1]  # 闭合图形

        ax.plot(angles, values, 'o-', linewidth=2, label=category)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(experts)
    ax.set_ylim(0, max(max([max(data[cat][e] for e in experts) for cat in categories]), 1))
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_analysis_results(expert_selections: Dict, question_types: Dict, output_dir: str):
    """保存分析结果"""
    results = {
        "summary": {
            "total_samples": sum(len(selections) for selections in expert_selections.values()),
            "num_routers": len(expert_selections),
            "question_type_distribution": dict(zip(*np.unique(list(question_types.values()), return_counts=True))),
        },
        "expert_selection_stats": {},
        "question_type_correlation": {}
    }

    # 统计专家选择情况
    for router_name, selections in expert_selections.items():
        if selections:
            results["expert_selection_stats"][router_name] = {
                "total_selections": len(selections),
                "avg_num_experts_per_selection": np.mean([len(s["experts"]) if "experts" in s else 0 for s in selections]),
            }

    # 保存结果
    with open(os.path.join(output_dir, "expert_analysis_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Expert Learning Content Analysis')
    parser.add_argument('--router_dump_path', type=str, required=True,
                        help='Path to the router dump JSONL file')
    parser.add_argument('--output_dir', type=str, default='./expert_analysis',
                        help='Output directory for analysis results')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行分析
    analyze_expert_selections(args.router_dump_path, args.output_dir)


if __name__ == "__main__":
    main()