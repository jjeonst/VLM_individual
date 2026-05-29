import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPTextModel

# ==========================================
# 1. 로봇의 뇌 구조 (Transformer Policy Network)
# ==========================================
class PR2LPolicy(nn.Module):
    def __init__(self, embed_dim=768, num_actions=4):
        super().__init__()
        # 트랜스포머 인코더 (그래프 노드들과 언어 목표의 관계를 파악)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 트랜스포머의 결과를 받아 최종 행동(Action ID)을 결정하는 출력층
        self.action_head = nn.Linear(embed_dim, num_actions)

    def forward(self, text_embed, graph_nodes):
        # 텍스트 벡터(1개)와 지도 노드 벡터(N개)를 합쳐서 하나의 문장(Sequence)처럼 만듭니다.
        # [Batch, 1+N, 768]
        combined_seq = torch.cat([text_embed.unsqueeze(1), graph_nodes], dim=1)
        
        # 트랜스포머 통과 (모든 노드와 텍스트가 서로 정보를 교환)
        out_seq = self.transformer(combined_seq)
        
        # 목표 언어 토큰(0번째 위치)의 최종 결과물만 뽑아서 행동 결정에 사용
        goal_features = out_seq[:, 0, :]
        action_logits = self.action_head(goal_features)
        return action_logits

# ==========================================
# 2. 파이토치 데이터셋 (하드디스크에서 데이터 공수)
# ==========================================
class HabitatDataset(Dataset):
    def __init__(self, data_dir, text_model, processor, device):
        self.data_dir = data_dir
        # _graph_nodes.npy 파일 목록만 추립니다.
        self.files = [f for f in os.listdir(data_dir) if f.endswith("_graph_nodes.npy")]
        
        # 💡 [핵심] 실제로는 에피소드마다 목표(json)가 다르지만, 여기서는 예시로 매핑합니다.
        # 실전에서는 collect_data에서 저장한 info.json 등을 불러와야 합니다.
        self.dummy_goals = ["Find a toilet", "Find a bed", "Find a sofa"]
        
        self.text_model = text_model
        self.processor = processor
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        ep_id = file_name.split("_")[0] # 예: 'ep_0'
        
        # 1. 위상 그래프 노드 불러오기
        graph_path = os.path.join(self.data_dir, file_name)
        graph_nodes = np.load(graph_path) # [N, 768]
        
        # 2. 정답 행동(Action) 불러오기
        action_path = os.path.join(self.data_dir, f"{ep_id}_actions.npy")
        actions = np.load(action_path)
        # 학습을 위해 해당 에피소드의 마지막 행동(또는 대표 행동)을 가져옵니다. 
        # (실제 Behavior Cloning에서는 Sequence-to-Sequence로 짜지만, 여기선 단순화)
        target_action = actions[-1] 

        # 3. 텍스트 목표를 실시간으로 임베딩
        target_text = self.dummy_goals[idx % len(self.dummy_goals)]
        inputs = self.processor(text=[target_text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embed = self.text_model(**inputs).pooler_output.cpu().numpy()[0] # [768]

        return torch.tensor(text_embed, dtype=torch.float32), \
               torch.tensor(graph_nodes, dtype=torch.float32), \
               torch.tensor(target_action, dtype=torch.long)

# 💡 길이가 제각각인 그래프 노드들을 똑같은 길이로 맞춰주는 패딩 함수
def collate_fn(batch):
    text_embeds, graph_nodes_list, actions = zip(*batch)
    text_embeds = torch.stack(text_embeds)
    actions = torch.stack(actions)
    
    # 노드 개수가 5개인 집, 10개인 집 등 다르기 때문에 pad_sequence로 빈칸을 0으로 채웁니다.
    graph_nodes_padded = torch.nn.utils.rnn.pad_sequence(graph_nodes_list, batch_first=True)
    return text_embeds, graph_nodes_padded, actions

# ==========================================
# 3. 진짜 학습 루프 (Train Loop)
# ==========================================
def main():
    print("🚀 PR2L 실전 학습 파이프라인을 가동합니다!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 언어 모델 세팅 (텍스트 임베딩용, 학습 안 시키므로 Frozen)
    print("🧠 언어 모델 로딩 중...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
    text_model = CLIPTextModel.from_pretrained(model_id).to(device)
    text_model.eval()

    # 데이터셋 & 데이터로더 세팅
    data_dir = "offline_data"
    dataset = HabitatDataset(data_dir, text_model, processor, device)
    
    if len(dataset) == 0:
        print("❌ 학습할 데이터가 없습니다. 5단계를 확인하세요!")
        return

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # 딥러닝 모델 & 최적화 도구 세팅
    model = PR2LPolicy(embed_dim=768, num_actions=4).to(device) # 로봇이 할 수 있는 행동 4가지 가정
    criterion = nn.CrossEntropyLoss() # 행동 예측은 '분류(Classification)' 문제이므로 CrossEntropy 사용
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 학습 시작 (Epoch)
    epochs = 50
    print("\n🔥 트랜스포머 학습을 시작합니다...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for text_embed, graph_nodes, target_action in dataloader:
            # 데이터를 GPU로 이동
            text_embed = text_embed.to(device)
            graph_nodes = graph_nodes.to(device)
            target_action = target_action.to(device)

            # 1. 현재 로봇의 예측값
            pred_logits = model(text_embed, graph_nodes)

            # 2. 오차(Loss) 계산 : "네가 예측한 행동과 전문가의 행동이 얼마나 다르니?"
            loss = criterion(pred_logits, target_action)

            # 3. 오차를 바탕으로 뇌 세포(가중치) 업데이트 (Backpropagation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        print(f"📈 Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # 학습된 뇌 저장
    torch.save(model.state_dict(), "pr2l_policy_best.pth")
    print("\n🎉 학습 완료! 로봇의 뇌 가중치가 'pr2l_policy_best.pth'로 저장되었습니다.")

if __name__ == "__main__":
    main()