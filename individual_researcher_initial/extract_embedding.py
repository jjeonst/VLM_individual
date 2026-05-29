import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from tqdm import tqdm # 진행률 바를 보여주기 위한 라이브러리 (pip install tqdm 필요할 수 있음)

def main():
    print("🖥️ GPU 상태를 점검합니다...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU(CUDA)를 찾을 수 없습니다!")
    device = "cuda"
    print(f"✅ GPU 활성화 완료: {torch.cuda.get_device_name(0)}")

    # VLM 로드 (경고 제거 적용)
    print("\n🧠 VLM 모델(CLIP Vision)을 로딩 중입니다...")
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
    model = CLIPVisionModel.from_pretrained(model_id).to(device)

    # 데이터 폴더 탐색
    data_dir = "offline_data"
    npy_files = [f for f in os.listdir(data_dir) if f.endswith("_rgb.npy")]
    
    if not npy_files:
        print("❌ 변환할 데이터가 없습니다.")
        return

    print(f"\n📂 총 {len(npy_files)}개의 에피소드 데이터를 변환합니다.")

    # GPU 메모리 터짐 방지를 위해 한 번에 처리할 사진 장수 (그래픽카드 성능에 따라 조절)
    BATCH_SIZE = 64 

    # 모든 파일 순회
    for file_name in npy_files:
        target_file = os.path.join(data_dir, file_name)
        rgb_frames = np.load(target_file)
        num_frames = len(rgb_frames)
        
        print(f"\n👉 [{file_name}] 처리 중... (총 {num_frames} 프레임)")
        
        all_embeddings = []
        
        # 데이터를 BATCH_SIZE 만큼 쪼개서 GPU에 올림
        for i in tqdm(range(0, num_frames, BATCH_SIZE)):
            batch_frames = rgb_frames[i : i + BATCH_SIZE]
            
            # 파이썬 이미지(PIL) 리스트로 변환
            images = [Image.fromarray(frame.astype('uint8'), mode='RGB') for frame in batch_frames]
            
            # 모델에 넣을 수 있게 전처리 후 GPU로 전송
            inputs = processor(images=images, return_tensors="pt").to(device)
            
            # 변환 수행
            with torch.no_grad():
                outputs = model(**inputs)
                # 추출된 임베딩을 다시 CPU로 내리고 Numpy 배열로 변환 (GPU 메모리 절약)
                embeds = outputs.pooler_output.cpu().numpy() 
                
            all_embeddings.append(embeds)
            
        # 쪼개서 처리한 배치들을 하나의 거대한 배열로 합치기
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # 저장 파일명 만들기 (예: ep_0_rgb.npy -> ep_0_embeds.npy)
        save_name = file_name.replace("_rgb.npy", "_embeds.npy")
        save_path = os.path.join(data_dir, save_name)
        
        np.save(save_path, final_embeddings)
        print(f"✅ 저장 완료: {save_path} (모양: {final_embeddings.shape})")

    print("\n🎉 모든 데이터의 VLM 임베딩 추출이 완벽하게 끝났습니다!")

if __name__ == "__main__":
    main()