import requests
from bs4 import BeautifulSoup
import re
import json
import os
from collections import defaultdict

def scrape_author_index(url, year):
    """指定したURLから著者索引をスクレイピングする"""
    print(f"Scraping {url} for year {year}...")
    
    response = requests.get(url)
    response.encoding = 'utf-8'  # 日本語対応
    
    if response.status_code != 200:
        print(f"Failed to fetch the page: {response.status_code}")
        return {}
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 著者索引の部分を取得
    author_index_text = ""
    author_index_section = soup.find(lambda tag: tag.name == 'h2' and '著者索引' in tag.text)
    
    if author_index_section:
        # 著者索引セクションの後のテキストを取得
        next_element = author_index_section.find_next()
        while next_element and next_element.name != 'h2':
            if next_element.name:
                author_index_text += next_element.get_text() + "\n"
            next_element = next_element.find_next()
    else:
        # 見出しが見つからない場合、ページ内の「著者索引」という文字列を含む部分を探す
        index_content = soup.find(text=re.compile('著者索引'))
        if index_content:
            # その要素の親要素から後のテキストを取得
            parent = index_content.parent
            next_element = parent
            while next_element:
                if next_element.name:
                    author_index_text += next_element.get_text() + "\n"
                next_element = next_element.find_next_sibling()
    
    if not author_index_text:
        print("Could not find author index section.")
        return {}
    
    # 著者と論文IDのペアを抽出
    # パターン：著者名に続いて論文IDがある（例：相澤 彰子P1-10, Q4-2, Q8-7）
    authors_data = {}
    
    # 「ア行」「カ行」などの見出しパターン
    heading_pattern = re.compile(r'【[ア-ン]行】')
    # 「アイ」「カキ」などの50音インデックスパターン
    index_pattern = re.compile(r'^[ア-ン]{1,2}\t')
    
    # 行ごとに処理
    for line in author_index_text.split('\n'):
        # 空行をスキップ
        if not line.strip():
            continue
        
        # 「【ア行】」などの見出し行はスキップ
        if heading_pattern.search(line):
            continue
        
        # 「アイ」などの50音インデックスを削除
        line = index_pattern.sub('', line)
        
        # 複数の著者を含む行を分割
        authors_in_line = re.split(r'　　', line)
        
        for author_entry in authors_in_line:
            # 著者名と論文IDを分離
            match = re.match(r'(.+?)\t([A-Z][0-9]-[0-9]+.*)', author_entry)
            if match:
                author_name = match.group(1).strip()
                paper_ids_text = match.group(2)
                
                # 論文IDを抽出
                paper_ids = re.findall(r'([A-Z][0-9]-[0-9]+)○?', paper_ids_text)
                
                # 主著者フラグを確認
                is_primary_author = []
                for paper_id in paper_ids:
                    if re.search(f'{paper_id}○', paper_ids_text):
                        is_primary_author.append(True)
                    else:
                        is_primary_author.append(False)
                
                authors_data[author_name] = {
                    'papers': paper_ids,
                    'is_primary': is_primary_author,
                    'year': year
                }
    
    return authors_data

def build_coauthor_network(authors_data):
    """著者データから共著者ネットワークを構築する"""
    # 論文ごとの著者リスト
    papers_authors = defaultdict(list)
    
    # 各著者の論文をマッピング
    for author, data in authors_data.items():
        for paper in data['papers']:
            papers_authors[paper].append(author)
    
    # 共著者関係を構築
    coauthor_network = defaultdict(list)
    
    for paper, authors in papers_authors.items():
        if len(authors) > 1:  # 共著論文の場合
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    # 双方向の関係を追加
                    if author2 not in coauthor_network[author1]:
                        coauthor_network[author1].append(author2)
                    if author1 not in coauthor_network[author2]:
                        coauthor_network[author2].append(author1)
    
    return dict(coauthor_network)

def create_d3_json(authors_data_all_years, coauthor_networks_all_years):
    """D3.js用のJSONデータを作成する"""
    # ノード（著者）のリスト
    nodes = []
    all_authors = set()
    
    # 各年のデータを統合
    for year, authors_data in authors_data_all_years.items():
        for author, data in authors_data.items():
            all_authors.add(author)
    
    # ノードを作成
    for author in all_authors:
        author_years = {}
        for year, authors_data in authors_data_all_years.items():
            if author in authors_data:
                author_years[year] = {
                    'papers_count': len(authors_data[author]['papers']),
                    'primary_count': sum(authors_data[author]['is_primary'])
                }
        
        nodes.append({
            'id': author,
            'name': author,
            'years': author_years
        })
    
    # リンク（共著関係）のリスト
    links = []
    
    # 各年の共著関係を統合
    for year, coauthor_network in coauthor_networks_all_years.items():
        for author1, coauthors in coauthor_network.items():
            for author2 in coauthors:
                # 重複を避けるために著者名でソート
                source, target = sorted([author1, author2])
                
                # 既存のリンクを探す
                existing_link = None
                for link in links:
                    if link['source'] == source and link['target'] == target:
                        existing_link = link
                        break
                
                if existing_link:
                    # 既存のリンクに年度を追加
                    if year not in existing_link['years']:
                        existing_link['years'].append(year)
                else:
                    # 新しいリンクを作成
                    links.append({
                        'source': source,
                        'target': target,
                        'years': [year]
                    })
    
    return {
        'nodes': nodes,
        'links': links
    }

def main():
    # 処理する年度とURL
    urls = {
        '2023': 'https://www.anlp.jp/proceedings/annual_meeting/2023/',
        '2024': 'https://www.anlp.jp/proceedings/annual_meeting/2024/',
        '2025': 'https://www.anlp.jp/proceedings/annual_meeting/2025/'
    }
    
    authors_data_all_years = {}
    coauthor_networks_all_years = {}
    
    # 各年度のデータをスクレイピング
    for year, url in urls.items():
        print(f"Processing year {year}...")
        authors_data = scrape_author_index(url, year)
        
        if authors_data:
            authors_data_all_years[year] = authors_data
            coauthor_networks_all_years[year] = build_coauthor_network(authors_data)
            print(f"Found {len(authors_data)} authors and {sum(len(coauthors) for coauthors in coauthor_networks_all_years[year].values()) // 2} coauthor relationships for {year}.")
        else:
            print(f"No data found for {year}.")
    
    # D3.js用のJSONデータを作成
    d3_data = create_d3_json(authors_data_all_years, coauthor_networks_all_years)
    
    # JSONファイルに保存
    with open('anlp_coauthor_network.json', 'w', encoding='utf-8') as f:
        json.dump(d3_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully created JSON file with {len(d3_data['nodes'])} nodes and {len(d3_data['links'])} links.")

if __name__ == "__main__":
    main()