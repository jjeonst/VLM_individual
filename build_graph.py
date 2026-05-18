import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def main():
    print("🗺️ 실제 VLM 임베딩 데이터를 기반으로 위상 그래프를 생성합니다...")

    # 1. 진짜 데이터 파일 로드
    data_dir = "offline_data"
    embed_files = [f for f in os.listdir(data_dir) if f.endswith("_embeds.npy")]
    
    if not embed_files:
        print("❌ offline_data 폴더에 '_embeds.npy' 파일이 없습니다. 4단계를 먼저 완료해주세요!")
        return

    # 우선 첫 번째 에피소드의 데이터로 지도를 그려봅니다.
    target_file = os.path.join(data_dir, embed_files[0])
    print(f"👉 로드할 파일: {target_file}")
    
    # 임베딩 데이터 불러오기 (예: 150프레임, 768차원)
    embeddings = np.load(target_file)
    num_frames = len(embeddings)
    print(f"👉 총 {num_frames}장의 사진(임베딩)을 분석하여 지도를 압축합니다.")

    # 2. 빈 그래프 도화지 준비
    G = nx.Graph()
    
    # 3. 임베딩 유사도 기준 설정
    # VLM 모델마다 출력값의 분포가 다릅니다. CLIP의 경우 보통 0.85~0.95 사이를 임계값으로 둡니다.
    # 만약 노드가 너무 안 묶이면 이 값을 낮추고, 너무 하나로 뭉치면 값을 높여주세요.
    threshold = 0.90  
    
    current_node_id = 0
    G.add_node(current_node_id, frames=[0]) 
    node_features = [embeddings[0]] # 0번 장소의 대표 얼굴(임베딩)
    
    print("\n⚡ 코사인 유사도를 계산하며 같은 장소들을 묶어냅니다...")
    for i in range(1, num_frames):
        # 현재 프레임 임베딩 vs 현재 장소의 대표 임베딩 비교
        sim = cosine_similarity([embeddings[i]], [node_features[current_node_id]])[0][0]
        
        if sim >= threshold:
            # 90% 이상 비슷함! -> 아직 같은 방 안에 있음 (노드에 프레임 추가)
            G.nodes[current_node_id]['frames'].append(i)
        else:
            # 풍경이 크게 달라짐! -> 새로운 방(노드)으로 진입함
            new_node_id = current_node_id + 1
            G.add_node(new_node_id, frames=[i])
            
            # 이전 방과 새로운 방 사이에 이동 가능함(Edge) 표시
            G.add_edge(current_node_id, new_node_id) 
            
            # 새로운 방의 대표 임베딩 저장
            node_features.append(embeddings[i])
            current_node_id = new_node_id
            
    # 4. 결과 요약 출력
    print("\n✅ 실제 데이터 기반 그래프 완성!")
    print(f"👉 {num_frames}장의 프레임이 총 {G.number_of_nodes()}개의 핵심 장소(Node)로 압축되었습니다.")
    
    for node in G.nodes():
        frames = G.nodes[node]['frames']
        print(f"  - Node {node}: 프레임 {frames[0]} ~ {frames[-1]} 포함 (총 {len(frames)}장 압축됨)")

    # (옵션) 정책 네트워크(Policy Network) 학습을 위해 압축된 노드들의 대표 임베딩만 따로 저장
    compressed_embeddings = np.array(node_features)
    save_npy_path = target_file.replace("_embeds.npy", "_graph_nodes.npy")
    np.save(save_npy_path, compressed_embeddings)
    print(f"\n💾 압축된 노드 데이터가 저장되었습니다: {save_npy_path}")

    # 5. 시각화 및 이미지 저장
    plt.figure(figsize=(10, 8))
    
    # 노드가 1자로만 그려지지 않도록 kamada_kawai 레이아웃 사용 (더 예쁘게 그려짐)
    pos = nx.kamada_kawai_layout(G) 
    
    # 프레임이 많이 묶인 장소일수록(오래 머문 곳) 원을 크게 그림
    node_sizes = [len(G.nodes[n]['frames']) * 100 for n in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=node_sizes, font_weight='bold', edge_color='gray', width=2)
    
    plt.title(f"Topological Map (Based on Real VLM Embeddings)\nThreshold: {threshold}", fontsize=14)
    
    save_img_path = target_file.replace("_embeds.npy", "_graph.png")
    plt.savefig(save_img_path)
    print(f"🎨 완성된 로봇의 실제 지도가 '{save_img_path}' 이미지로 저장되었습니다!")

if __name__ == "__main__":
    main()