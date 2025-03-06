import re
import json
import numpy as np
import networkx as nx
from collections import defaultdict
from test import generate_testdata

def parse_author_index(text):
    """著者索引のテキストからデータを抽出する"""
    lines = text.strip().split('\n')
    authors_data = {}
    
    # 「ア行」「カ行」などの見出しパターン
    heading_pattern = re.compile(r'【[ア-ン]行】')
    # 「アイ」「カキ」などの50音インデックスパターン
    index_pattern = re.compile(r'^([ア-ン]{1,2})\t')
    
    current_heading = None
    current_index = None
    
    for line in lines:
        # 空行をスキップ
        if not line.strip():
            continue
        
        # 「【ア行】」などの見出し行を処理
        heading_match = heading_pattern.search(line)
        if heading_match:
            current_heading = line.strip()
            continue
        
        # 「アイ」などの50音インデックス行を処理
        index_match = index_pattern.match(line)
        if index_match:
            current_index = index_match.group(1)
            # 残りの部分を処理
            line = index_pattern.sub('', line)
        
        # 複数の著者を含む行を分割
        authors_in_line = re.split(r'　　', line)
        
        for author_entry in authors_in_line:
            # 著者名と論文IDを分離
            match = re.match(r'(.+?)\t(.+)', author_entry)
            
            if match:
                author_name = match.group(1).strip()
                paper_ids_text = match.group(2).strip()
                
                # 論文IDを抽出
                paper_ids = re.findall(r'([A-Z][0-9]-[0-9]+)○?', paper_ids_text)
                
                # 主著者フラグを確認
                is_primary_author = {}
                for paper_id in paper_ids:
                    is_primary = '○' in paper_ids_text and f'{paper_id}○' in paper_ids_text
                    is_primary_author[paper_id] = is_primary
                
                authors_data[author_name] = {
                    'heading': current_heading,
                    'index': current_index,
                    'papers': paper_ids,
                    'is_primary': is_primary_author
                }
    
    return authors_data

def create_papers_authors_mapping(authors_data):
    """論文IDごとの著者リストを作成"""
    papers_mapping = defaultdict(list)
    
    for author_name, data in authors_data.items():
        for paper_id in data['papers']:
            is_primary = data['is_primary'].get(paper_id, False)
            papers_mapping[paper_id].append({
                'name': author_name,
                'is_primary': is_primary
            })
    
    return dict(papers_mapping)

def create_coauthor_network(papers_data):
    """共著関係ネットワークを作成"""
    coauthor_network = defaultdict(list)
    
    for paper_id, authors in papers_data.items():
        if len(authors) > 1:  # 共著論文の場合
            author_names = [a['name'] for a in authors]
            for i, author1 in enumerate(author_names):
                for author2 in author_names[i+1:]:
                    if author2 not in coauthor_network[author1]:
                        coauthor_network[author1].append(author2)
                    if author1 not in coauthor_network[author2]:
                        coauthor_network[author2].append(author1)
    
    return dict(coauthor_network)

def calculate_author_metrics(authors_data, papers_data, coauthor_network):
    """著者の各種指標を計算"""
    # NetworkXグラフの構築
    G = nx.Graph()
    
    # 著者をノードとして追加
    for author_name in authors_data.keys():
        G.add_node(author_name)
    
    # 共著関係をエッジとして追加
    for author1, coauthors in coauthor_network.items():
        for author2 in coauthors:
            # 共著した論文数をカウント
            shared_papers = []
            for paper_id, authors in papers_data.items():
                author_names = [a['name'] for a in authors]
                if author1 in author_names and author2 in author_names:
                    shared_papers.append(paper_id)
            
            G.add_edge(author1, author2, weight=len(shared_papers))
    
    # 各種中心性指標の計算
    # PageRank
    pagerank = nx.pagerank(G, weight='weight')
    
    # 次数中心性
    degree_centrality = nx.degree_centrality(G)
    
    # 媒介中心性
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    
    # 近接中心性
    closeness_centrality = nx.closeness_centrality(G, distance='weight')
    
    # コミュニティ検出（Louvainアルゴリズム）
    try:
        import community as community_louvain
        communities = community_louvain.best_partition(G)
    except ImportError:
        # python-louvainがない場合はNetworkXの内蔵関数を使用
        from networkx.algorithms import community
        communities_generator = community.greedy_modularity_communities(G)
        communities = {}
        for i, comm in enumerate(communities_generator):
            for node in comm:
                communities[node] = i
                
    # 著者情報に指標を追加
    author_metrics = {}
    for author_name in authors_data.keys():
        papers_count = len(authors_data[author_name]['papers'])
        primary_count = sum(1 for is_primary in authors_data[author_name]['is_primary'].values() if is_primary)
        coauthors_count = len(coauthor_network.get(author_name, []))
        
        author_metrics[author_name] = {
            'papers_count': papers_count,
            'primary_count': primary_count,
            'coauthors_count': coauthors_count,
            'pagerank': pagerank.get(author_name, 0),
            'degree_centrality': degree_centrality.get(author_name, 0),
            'betweenness_centrality': betweenness_centrality.get(author_name, 0),
            'closeness_centrality': closeness_centrality.get(author_name, 0),
            'community': communities.get(author_name, 0)
        }
    
    return author_metrics

def create_enhanced_d3_json(authors_data, papers_data, coauthor_network, author_metrics):
    """拡張されたD3.js用のJSONデータを作成"""
    nodes = []
    links = []
    
    # ノード（著者）の作成
    for author_name, data in authors_data.items():
        metrics = author_metrics[author_name]
        
        nodes.append({
            'id': author_name,
            'name': author_name,
            'heading': data['heading'],
            'index': data['index'],
            'papers_count': metrics['papers_count'],
            'primary_count': metrics['primary_count'],
            'coauthors_count': metrics['coauthors_count'],
            'pagerank': metrics['pagerank'],
            'degree_centrality': metrics['degree_centrality'],
            'betweenness_centrality': metrics['betweenness_centrality'],
            'closeness_centrality': metrics['closeness_centrality'],
            'community': metrics['community'],
            'papers': data['papers'],
            # 主著者かどうかの情報を保持
            'primary_papers': [p for p, is_primary in data['is_primary'].items() if is_primary]
        })
    
    # リンク（共著関係）の作成
    for author1, coauthors in coauthor_network.items():
        for author2 in coauthors:
            # 共著した論文数をカウント
            shared_papers = []
            for paper_id, authors in papers_data.items():
                author_names = [a['name'] for a in authors]
                if author1 in author_names and author2 in author_names:
                    shared_papers.append(paper_id)
            
            links.append({
                'source': author1,
                'target': author2,
                'value': len(shared_papers),
                'papers': shared_papers
            })
    
    return {
        'nodes': nodes,
        'links': links
    }

def main():
    # テスト用の著者索引データ
    test_data = generate_testdata()
    
    # 解析
    authors_data = parse_author_index(test_data)
    
    # 論文-著者マッピング
    papers_data = create_papers_authors_mapping(authors_data)
    
    # 共著ネットワーク
    coauthor_network = create_coauthor_network(papers_data)
    
    # 著者指標の計算
    author_metrics = calculate_author_metrics(authors_data, papers_data, coauthor_network)
    
    # 拡張されたD3.js用のJSONデータ作成
    d3_data = create_enhanced_d3_json(authors_data, papers_data, coauthor_network, author_metrics)
    
    # JSONファイルに保存
    with open('enhanced_anlp_author_network.json', 'w', encoding='utf-8') as f:
        json.dump(d3_data, f, ensure_ascii=False, indent=2)
    
    print(f"拡張JSONファイルを作成しました: {len(d3_data['nodes'])}ノード、{len(d3_data['links'])}リンク")
    
    # 結果表示
    print("\n===== 著者指標計算結果 =====")
    for author, metrics in sorted(author_metrics.items(), key=lambda x: x[1]['pagerank'], reverse=True):
        print(f"{author}:")
        print(f"  論文数: {metrics['papers_count']}")
        print(f"  主著者数: {metrics['primary_count']}")
        print(f"  共著者数: {metrics['coauthors_count']}")
        print(f"  PageRank: {metrics['pagerank']:.6f}")
        print(f"  次数中心性: {metrics['degree_centrality']:.6f}")
        print(f"  媒介中心性: {metrics['betweenness_centrality']:.6f}")
        print(f"  近接中心性: {metrics['closeness_centrality']:.6f}")
        print(f"  コミュニティ: {metrics['community']}")
        print()

if __name__ == "__main__":
    main()