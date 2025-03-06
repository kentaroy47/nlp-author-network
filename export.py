import requests
from bs4 import BeautifulSoup
import networkx as nx
import pandas as pd
import re
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Python-Louvainパッケージのインポート
# 新しいバージョンでは python-louvain パッケージをインストールして
# import community.community_louvain as community_louvain を使用
try:
    import community.community_louvain as community_louvain
except ImportError:
    # 古いバージョンの場合
    try:
        import community as community_louvain
    except ImportError:
        print("コミュニティ検出ライブラリがインストールされていません。")
        print("以下のコマンドでインストールしてください:")
        print("pip install python-louvain networkx pandas matplotlib numpy beautifulsoup4 requests")
        community_louvain = None

# --- 1. データ収集部分 ---
def extract_authors(authors_text):
    """著者名のみを抽出する改良版関数"""
    # 前処理：余分な空白、改行、タブを削除
    authors_text = authors_text.strip().replace('\n', ' ').replace('\t', ' ')
    
    # 一般的な区切り文字でスプリット
    separators = [',', '，', '、', '・', ';', '；']
    
    # どの区切り文字が最も多く使われているかを確認
    best_separator = None
    max_count = 0
    
    for sep in separators:
        count = authors_text.count(sep)
        if count > max_count:
            max_count = count
            best_separator = sep
    
    if best_separator and max_count > 0:
        # 最適な区切り文字で分割
        raw_authors = authors_text.split(best_separator)
    else:
        # 区切り文字がない場合は空白で分割を試みる
        raw_authors = authors_text.split()
    
    # 著者名のクリーニング
    authors = []
    for author in raw_authors:
        # 前後の空白を削除
        author = author.strip()
        
        # 一般的な役職や所属を示す単語をフィルタリング
        ignore_terms = ['教授', '准教授', '助教', '研究員', '大学院生', '博士', 'Prof.', 'Dr.', 
                         'University', '大学', '研究所', 'Institute', '株式会社', 'Co., Ltd.']
        
        has_ignore_term = any(term in author for term in ignore_terms)
        
        # 著者名として不適切な長さのものを除外（あまりに短すぎる/長すぎる）
        is_valid_length = 2 <= len(author) <= 20
        
        # 数字や特殊文字が多すぎる場合は除外
        special_chars = sum(1 for c in author if not c.isalpha() and not c.isspace())
        has_many_special = special_chars > len(author) / 3
        
        if is_valid_length and not has_ignore_term and not has_many_special:
            # 括弧内の情報を削除（所属や肩書きが入っていることが多い）
            clean_author = re.sub(r'\([^)]*\)', '', author)
            clean_author = re.sub(r'（[^）]*）', '', clean_author)
            
            # 前後の空白を再度削除
            clean_author = clean_author.strip()
            
            if clean_author:
                authors.append(clean_author)
    
    return authors

def scrape_paper_data(url, year):
    """ウェブページから論文と著者情報を抽出する"""
    print(f"{year}年のデータを収集中: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"データ取得エラー: {e}")
        return []
    
    papers = []
    
    # 論文タイトルと著者の特定のパターンを探す
    paper_elements = soup.find_all(['h2', 'h3', 'h4'], class_=['title', 'paper-title'])
    for element in paper_elements:
        title = element.text.strip()
        # 著者情報は通常タイトルの後に来る
        authors_element = element.find_next(['p', 'div'], class_=['authors', 'paper-authors'])
        if authors_element:
            authors_text = authors_element.text.strip()
            # 改良した関数で著者のみを抽出
            authors = extract_authors(authors_text)
            if authors:  # 著者が抽出できた場合のみ追加
                papers.append({'title': title, 'authors': authors, 'year': year})
    
    # 他の抽出方法も同様に改善...
    
    return papers

# --- 2. ネットワーク構築部分 ---
def build_author_network(papers, year=None):
    """論文データから著者ネットワークを構築する"""
    author_links = defaultdict(int)
    all_authors = set()
    author_papers = defaultdict(list)  # 著者ごとの論文リスト
    
    # 特定の年のみフィルタリング
    if year:
        papers = [p for p in papers if p.get('year') == year]
    
    for i, paper in enumerate(papers):
        authors = paper['authors']
        
        # 著者名が明らかに不適切な場合はスキップ
        if any(len(author) > 30 for author in authors) or len(authors) > 15:
            continue
            
        all_authors.update(authors)
        
        # 著者と論文のマッピング
        for author in authors:
            author_papers[author].append(i)
        
        # 共著関係の構築
        for i in range(len(authors)):
            for j in range(i+1, len(authors)):
                if authors[i] < authors[j]:
                    author_links[(authors[i], authors[j])] += 1
                else:
                    author_links[(authors[j], authors[i])] += 1
    
    # NetworkXグラフの構築
    G = nx.Graph()
    
    # ノード追加と属性設定
    for author in all_authors:
        paper_count = len(author_papers[author])
        G.add_node(author, papers=paper_count, paper_ids=author_papers[author])
    
    # エッジ追加
    for (author1, author2), weight in author_links.items():
        G.add_edge(author1, author2, weight=weight)
    
    return G, papers

# --- 3. ネットワーク解析部分 ---
def analyze_network(G):
    """ネットワークの中心性指標やコミュニティを分析する"""
    results = {}
    
    # 次数中心性 (単純な共著数)
    degree_centrality = nx.degree_centrality(G)
    
    # 媒介中心性 (ブリッジとなる著者)
    betweenness_centrality = nx.betweenness_centrality(G, k=None, normalized=True)
    
    # 固有ベクトル中心性 (影響力の強い著者とのつながり)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        print("固有ベクトル中心性の計算に失敗しました。ノード間の接続が疎である可能性があります。")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()}
    
    # PageRank (重み付き影響力)
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # コミュニティ検出 (Louvain法)
    if community_louvain is not None:
        try:
            communities = community_louvain.best_partition(G)
        except:
            print("Louvainコミュニティ検出に失敗しました。代替方法を使用します。")
            # 連結成分をコミュニティとして使用
            communities = {}
            for i, comp in enumerate(nx.connected_components(G)):
                for node in comp:
                    communities[node] = i
    else:
        # コミュニティ検出ライブラリがない場合は連結成分を使用
        print("コミュニティ検出ライブラリがないため、連結成分をコミュニティとして使用します。")
        communities = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                communities[node] = i
    
    # 結果の統合
    results['degree'] = degree_centrality
    results['betweenness'] = betweenness_centrality
    results['eigenvector'] = eigenvector_centrality
    results['pagerank'] = pagerank
    results['communities'] = communities
    
    # コミュニティごとの著者リスト作成
    community_groups = defaultdict(list)
    for author, community_id in communities.items():
        community_groups[community_id].append(author)
    
    return results, community_groups

# --- 4. データエクスポート部分 ---
def export_network_data(G, analysis_results, community_groups, papers, year=None, output_prefix="author_network"):
    """ネットワークデータを各種形式でエクスポートする"""
    if year:
        output_prefix = f"{output_prefix}_{year}"
    
    # 4.1 D3.js用のJSONエクスポート
    d3_data = {
        "nodes": [],
        "links": []
    }
    
    # ノード情報を追加
    for node in G.nodes():
        node_data = {
            "id": node,
            "name": node,
            "papers": G.nodes[node]['papers'],
            "degree": analysis_results['degree'][node],
            "betweenness": analysis_results['betweenness'][node],
            "eigenvector": analysis_results['eigenvector'][node],
            "pagerank": analysis_results['pagerank'][node],
            "community": analysis_results['communities'][node]
        }
        d3_data["nodes"].append(node_data)
    
    # エッジ情報を追加
    for source, target, data in G.edges(data=True):
        link_data = {
            "source": source,
            "target": target,
            "weight": data['weight']
        }
        d3_data["links"].append(link_data)
    
    # D3.js用JSONとして出力
    with open(f"{output_prefix}_d3.json", 'w', encoding='utf-8') as f:
        json.dump(d3_data, f, ensure_ascii=False, indent=2)
    
    # 4.2 Gephi用のCSVエクスポート
    # ノードCSV
    nodes_df = pd.DataFrame([
        {
            "Id": node,
            "Label": node,
            "Papers": G.nodes[node]['papers'],
            "Degree": analysis_results['degree'][node],
            "Betweenness": analysis_results['betweenness'][node],
            "Eigenvector": analysis_results['eigenvector'][node],
            "PageRank": analysis_results['pagerank'][node],
            "Community": analysis_results['communities'][node]
        } for node in G.nodes()
    ])
    nodes_df.to_csv(f"{output_prefix}_nodes.csv", index=False, encoding='utf-8')
    
    # エッジCSV
    edges_df = pd.DataFrame([
        {
            "Source": source,
            "Target": target,
            "Weight": data['weight'],
            "Type": "Undirected"
        } for source, target, data in G.edges(data=True)
    ])
    edges_df.to_csv(f"{output_prefix}_edges.csv", index=False, encoding='utf-8')
    
    # 4.3 コミュニティ情報のエクスポート
    community_info = []
    for comm_id, authors in community_groups.items():
        community_size = len(authors)
        avg_papers = np.mean([G.nodes[author]['papers'] for author in authors])
        top_authors = sorted(authors, key=lambda a: analysis_results['pagerank'][a], reverse=True)[:5]
        
        community_info.append({
            "Community_ID": comm_id,
            "Size": community_size,
            "Avg_Papers": avg_papers,
            "Top_Authors": ", ".join(top_authors)
        })
    
    comm_df = pd.DataFrame(community_info)
    comm_df.to_csv(f"{output_prefix}_communities.csv", index=False, encoding='utf-8')
    
    # 4.4 中心的著者ランキングのエクスポート
    centrality_df = pd.DataFrame([
        {
            "Author": author,
            "Papers": G.nodes[author]['papers'],
            "Degree": analysis_results['degree'][author],
            "Betweenness": analysis_results['betweenness'][author],
            "Eigenvector": analysis_results['eigenvector'][author],
            "PageRank": analysis_results['pagerank'][author],
            "Community": analysis_results['communities'][author]
        } for author in G.nodes()
    ])
    
    # PageRankでソート（最も総合的な重要度指標）
    centrality_df = centrality_df.sort_values('PageRank', ascending=False)
    centrality_df.to_csv(f"{output_prefix}_centrality.csv", index=False, encoding='utf-8')
    
    # 年度情報を含むオブジェクトを返す
    year_info = {
        "year": year,
        "nodes": len(G.nodes()),
        "links": len(G.edges()),
        "communities": len(community_groups),
        "top_authors": [{"name": row["Author"], "pagerank": row["PageRank"]} 
                        for _, row in centrality_df.head(10).iterrows()]
    }
    
    return d3_data, year_info

# --- 5. 簡易可視化部分 ---
def visualize_community_network(G, communities, output_file="community_network.png"):
    """コミュニティ情報を含むネットワークの簡易可視化"""
    plt.figure(figsize=(15, 15))
    
    # コミュニティごとに色分け
    colors = plt.cm.rainbow(np.linspace(0, 1, max(communities.values()) + 1))
    
    # ノードの色をコミュニティごとに設定
    node_colors = [colors[communities[node]] for node in G.nodes()]
    
    # ノードサイズを論文数に基づいて設定
    node_sizes = [100 * G.nodes[node]['papers'] for node in G.nodes()]
    
    # エッジの太さを共著回数に基づいて設定
    edge_widths = [G[u][v]['weight'] * 0.8 for u, v in G.edges()]
    
    # レイアウト設定
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    # ネットワーク描画
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray')
    
    # 重要なノードのラベルのみ表示
    pagerank = nx.pagerank(G)
    important_nodes = [node for node, pr in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:30]]
    nx.draw_networkx_labels(G, pos, {n: n for n in important_nodes}, font_size=10)
    
    plt.title('論文著者コミュニティネットワーク', fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# --- メイン実行部分 ---
def main():
    # 解析対象年度とURL
    years_urls = {
        2021: "https://www.anlp.jp/proceedings/annual_meeting/2021/",
        2022: "https://www.anlp.jp/proceedings/annual_meeting/2022/",
        2023: "https://www.anlp.jp/proceedings/annual_meeting/2023/",
        2024: "https://www.anlp.jp/proceedings/annual_meeting/2024/",
        2025: "https://www.anlp.jp/proceedings/annual_meeting/2025/"
    }
    
    # 出力ディレクトリ作成
    os.makedirs("data", exist_ok=True)
    
    all_papers = []
    years_info = []
    
    # 各年度のデータを収集
    for year, url in years_urls.items():
        papers = scrape_paper_data(url, year)
        all_papers.extend(papers)
        
        # サーバーに負荷をかけないように少し待機
        time.sleep(2)
    
    # 年度ごとのネットワーク分析
    for year in years_urls.keys():
        print(f"\n{year}年のネットワークを分析中...")
        G, year_papers = build_author_network(all_papers, year)
        
        if G.number_of_nodes() > 0:
            print(f"ネットワークノード数（著者数）: {G.number_of_nodes()}")
            print(f"ネットワークエッジ数（共著関係）: {G.number_of_edges()}")
            
            # ネットワーク解析
            analysis_results, community_groups = analyze_network(G)
            
            # データエクスポート
            _, year_info = export_network_data(
                G, analysis_results, community_groups, year_papers, 
                year=year, output_prefix="data/author_network"
            )
            
            years_info.append(year_info)
            
            # 簡易可視化
            visualize_community_network(
                G, analysis_results['communities'], 
                output_file=f"data/community_network_{year}.png"
            )
        else:
            print(f"{year}年のデータからネットワークを構築できませんでした。")
    
    # 全期間のネットワーク分析
    print("\n全期間のネットワークを分析中...")
    G_all, _ = build_author_network(all_papers)
    
    if G_all.number_of_nodes() > 0:
        print(f"全期間ネットワークノード数（著者数）: {G_all.number_of_nodes()}")
        print(f"全期間ネットワークエッジ数（共著関係）: {G_all.number_of_edges()}")
        
        # ネットワーク解析
        analysis_results, community_groups = analyze_network(G_all)
        
        # データエクスポート
        _, all_info = export_network_data(
            G_all, analysis_results, community_groups, all_papers, 
            output_prefix="data/author_network_all"
        )
        
        # 年度情報をインデックスファイルとして保存
        years_info.append({
            "year": "all",
            "nodes": all_info["nodes"],
            "links": all_info["links"],
            "communities": all_info["communities"],
            "top_authors": all_info["top_authors"]
        })
        
        # 簡易可視化
        visualize_community_network(
            G_all, analysis_results['communities'], 
            output_file="data/community_network_all.png"
        )
    else:
        print("全期間データからネットワークを構築できませんでした。")
    
    # 年度情報をJSONとして保存（D3.js用）
    with open("data/years_index.json", 'w', encoding='utf-8') as f:
        json.dump(years_info, f, ensure_ascii=False, indent=2)
    
    print("\n分析完了。以下のファイルが生成されました：")
    print("- data/years_index.json (年度インデックス)")
    print("- data/author_network_YEAR_d3.json (年度ごとのD3.js用データ)")
    print("- data/author_network_all_d3.json (全期間のD3.js用データ)")
    print("- data/community_network_YEAR.png (年度ごとの簡易可視化画像)")

if __name__ == "__main__":
    main()