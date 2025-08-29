import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, global_mean_pool
from rdkit import Chem
import numpy as np

# --- 1. SMILES를 그래프로 변환하는 유틸리티 함수 ---
# 논문의 2.4, 2.5 섹션에 해당하는 부분입니다.

def get_atom_features(atom):
    """원자(Node)의 특징(Feature)을 추출하여 벡터로 변환"""
    # 논문에서는 57차원 벡터를 사용했으나, 여기서는 핵심 특징만 간소화하여 구현
    # 원자 종류, 이웃 수소 원자 수, 형식 전하, 방향족 여부 등
    feature = np.zeros(14)
    feature[list(map(lambda s: s.GetSymbol(), Chem.GetPeriodicTable().GetValenceList(-1))).index(atom.GetSymbol())] = 1
    feature[10] = atom.GetTotalNumHs()
    feature[11] = atom.GetFormalCharge()
    feature[12] = atom.GetDegree()
    feature[13] = int(atom.IsInRing())
    return torch.tensor(feature, dtype=torch.float)

def get_bond_features(bond):
    """화학 결합(Edge)의 특징을 추출하여 벡터로 변환"""
    # 결합 종류 (단일, 이중, 삼중, 방향족)
    bond_type = bond.GetBondType()
    bond_map = {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
        Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1]
    }
    return torch.tensor(bond_map.get(bond_type, [0, 0, 0, 0]), dtype=torch.float)

def smiles_to_graph(smiles: str):
    """SMILES 문자열을 PyTorch Geometric 그래프 데이터 객체로 변환"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node 특징 (원자)
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)

    # Edge Index 및 Edge 특징 (결합)
    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_indices.append((j, i)) # 무방향 그래프
        
        bond_feat = get_bond_features(bond)
        edge_attrs.append(bond_feat)
        edge_attrs.append(bond_feat)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attrs)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# --- 2. 딥러닝 모델 정의 ---
# 논문의 2.7, 2.8 섹션 및 Figure 5, 6에 해당하는 부분입니다.

class SMP_Stress_Predictor(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, temporal_feature_dim,
                 gnn_hidden_dim, num_gnn_layers, num_heads,
                 transformer_hidden_dim, num_transformer_layers, window_size):
        super(SMP_Stress_Predictor, self).__init__()
        
        self.window_size = window_size
        self.gnn_hidden_dim = gnn_hidden_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        
        # Part 1: Graph Transformer (분자 구조 인코딩)
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(TransformerConv(node_feature_dim, gnn_hidden_dim, heads=num_heads, edge_dim=edge_feature_dim))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(TransformerConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, heads=num_heads, edge_dim=edge_feature_dim))

        # 동적 임베딩 생성을 위한 MLP
        self.dynamic_embedding_mlp = nn.Sequential(
            nn.Linear(gnn_hidden_dim * num_heads + (temporal_feature_dim * window_size), gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim)
        )

        # Part 2: Time Series Transformer (시계열 예측)
        # 입력 차원: 시계열 특징 + 그래프 임베딩
        encoder_input_dim = temporal_feature_dim + gnn_hidden_dim
        self.input_projection = nn.Linear(encoder_input_dim, transformer_hidden_dim)
        
        # PyTorch의 내장 Transformer 사용
        transformer_layer = nn.TransformerEncoderLayer(d_model=transformer_hidden_dim, nhead=num_heads, dim_feedforward=transformer_hidden_dim*4, batch_first=True)
        self.time_series_transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        
        # 최종 출력 레이어 (응력 값 1개 예측)
        self.output_layer = nn.Linear(transformer_hidden_dim, 1)

    def forward(self, epoxy_smiles_list, hardener_smiles_list, temporal_windows):
        # `temporal_windows` shape: (batch_size, window_size, temporal_features)
        
        batch_size = temporal_windows.shape[0]
        
        # --- Graph Embedding 생성 ---
        epoxy_graphs = [smiles_to_graph(s) for s in epoxy_smiles_list]
        hardener_graphs = [smiles_to_graph(s) for s in hardener_smiles_list]
        
        # 에폭시와 경화제 그래프를 하나의 배치로 결합 (Disconnected Graph)
        combined_graphs = [epoxy_graphs[i] for i in range(batch_size)] + [hardener_graphs[i] for i in range(batch_size)]
        graph_batch = Batch.from_data_list(combined_graphs)
        
        x, edge_index, edge_attr, batch_map = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch
        
        # Graph Transformer 연산
        for layer in self.gnn_layers:
            x = F.relu(layer(x, edge_index, edge_attr))
            
        # Global Pooling으로 각 분자 그래프의 대표 벡터 추출
        pooled_x = global_mean_pool(x, batch_map) # Shape: (2 * batch_size, gnn_hidden_dim * num_heads)
        
        # 에폭시와 경화제 임베딩을 결합 (여기서는 평균 사용)
        epoxy_embed = pooled_x[:batch_size]
        hardener_embed = pooled_x[batch_size:]
        static_graph_embedding = (epoxy_embed + hardener_embed) / 2 # Shape: (batch_size, gnn_hidden_dim * num_heads)

        # "동적" 그래프 임베딩 생성 (논문 아이디어)
        # 정적 그래프 임베딩에 시계열 윈도우 정보를 결합
        flattened_temporal = temporal_windows.view(batch_size, -1)
        dynamic_input = torch.cat([static_graph_embedding, flattened_temporal], dim=1)
        dynamic_graph_embedding = self.dynamic_embedding_mlp(dynamic_input) # Shape: (batch_size, gnn_hidden_dim)

        # --- Time Series 예측 ---
        # Time Series Transformer의 입력 준비
        # (batch, window_size, features) 형태로 만들기 위해 그래프 임베딩을 복제
        expanded_graph_embedding = dynamic_graph_embedding.unsqueeze(1).repeat(1, self.window_size, 1)
        
        # 시계열 데이터와 그래프 임베딩 결합
        transformer_input = torch.cat([temporal_windows, expanded_graph_embedding], dim=2)
        
        # Transformer 연산
        projected_input = self.input_projection(transformer_input)
        transformer_output = self.time_series_transformer(projected_input) # Shape: (batch_size, window_size, transformer_hidden_dim)

        # 마지막 타임스텝의 출력만 사용하여 최종 예측
        final_step_output = transformer_output[:, -1, :]
        prediction = self.output_layer(final_step_output)
        
        return prediction.squeeze(-1)

# --- 3. 학습 과정 실행 ---
if __name__ == '__main__':
    # --- 하이퍼파라미터 설정 ---
    WINDOW_SIZE = 4
    NODE_FEAT_DIM = 14 # get_atom_features 에서 정의
    EDGE_FEAT_DIM = 4 # get_bond_features 에서 정의
    TEMPORAL_FEAT_DIM = 3 # Time, Temperature, Length
    
    # 모델 파라미터
    GNN_HIDDEN = 32
    GNN_LAYERS = 2
    TRANSFORMER_HIDDEN = 64
    TRANSFORMER_LAYERS = 2
    NUM_HEADS = 4
    
    # 학습 파라미터
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    
    # --- 가상 데이터 생성 ---
    # 실제로는 파일에서 로드해야 합니다.
    print("🧪 가상 데이터 생성 중...")
    
    # 예제 SMILES
    epoxy_smiles = "CC(C)(C1=CC=C(C=C1)O[C@H]2COC2)C3=CC=C(C=C3)O[C@H]4COC4" # DGEBA
    hardener_smiles = "CC1(C(CC(C1)(C)CN)N)C" # IPD
    
    # 2개의 다른 조합을 시뮬레이션
    smiles_pairs = [
        (epoxy_smiles, hardener_smiles),
        ("C1=CC=C2C(=C1)C(=O)OC2(C3=CC=C(C=C3)O[C@H]4COC4)C5=CC=C(C=C5)O[C@H]6COC6", hardener_smiles) # 다른 에폭시
    ]
    
    # 시계열 데이터 생성 (1000 타임스텝)
    num_samples = len(smiles_pairs)
    total_timesteps = 1000
    
    all_temporal_data = torch.randn(num_samples, total_timesteps, TEMPORAL_FEAT_DIM)
    all_stress_data = torch.randn(num_samples, total_timesteps) # 정답 데이터
    
    # 슬라이딩 윈도우 데이터셋 생성
    inputs, labels, smiles_indices = [], [], []
    for i in range(num_samples):
        for t in range(total_timesteps - WINDOW_SIZE):
            inputs.append(all_temporal_data[i, t:t+WINDOW_SIZE, :])
            labels.append(all_stress_data[i, t+WINDOW_SIZE-1])
            smiles_indices.append(i)
    
    dataset_size = len(inputs)
    print(f"✅ 총 {dataset_size}개의 윈도우 샘플 생성 완료.")
    
    # --- 모델 및 학습 설정 ---
    model = SMP_Stress_Predictor(
        node_feature_dim=NODE_FEAT_DIM, edge_feature_dim=EDGE_FEAT_DIM, 
        temporal_feature_dim=TEMPORAL_FEAT_DIM,
        gnn_hidden_dim=GNN_HIDDEN, num_gnn_layers=GNN_LAYERS, num_heads=NUM_HEADS,
        transformer_hidden_dim=TRANSFORMER_HIDDEN, num_transformer_layers=TRANSFORMER_LAYERS,
        window_size=WINDOW_SIZE
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # --- 학습 루프 ---
    print("\n🚀 모델 학습 시작...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        
        # 데이터 순서를 섞어줌
        shuffled_indices = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, BATCH_SIZE):
            batch_indices = shuffled_indices[i:i+BATCH_SIZE]
            
            # 배치 데이터 준비
            batch_temporal = torch.stack([inputs[j] for j in batch_indices])
            batch_labels = torch.stack([labels[j] for j in batch_indices])
            
            batch_smiles_indices = [smiles_indices[j] for j in batch_indices]
            batch_epoxy_smiles = [smiles_pairs[k][0] for k in batch_smiles_indices]
            batch_hardener_smiles = [smiles_pairs[k][1] for k in batch_smiles_indices]

            # 예측 및 손실 계산
            optimizer.zero_grad()
            predictions = model(batch_epoxy_smiles, batch_hardener_smiles, batch_temporal)
            loss = criterion(predictions, batch_labels)
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / (dataset_size / BATCH_SIZE)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}")
            
    print("🎉 학습 완료!")
    
    # --- 테스트 (학습된 모델로 샘플 하나 예측) ---
    print("\n🔬 학습된 모델로 샘플 예측 테스트...")
    model.eval()
    with torch.no_grad():
        sample_temporal = inputs[0].unsqueeze(0) # 배치 차원 추가
        sample_epoxy = smiles_pairs[smiles_indices[0]][0]
        sample_hardener = smiles_pairs[smiles_indices[0]][1]
        
        prediction = model([sample_epoxy], [sample_hardener], sample_temporal)
        
        print(f"  - 샘플 Epoxy SMILES: {sample_epoxy[:30]}...")
        print(f"  - 샘플 Hardener SMILES: {sample_hardener[:30]}...")
        print(f"  - 입력 시계열 데이터 (마지막 스텝): {sample_temporal[0, -1, :].numpy()}")
        print("-" * 20)
        print(f"  🎯 실제 응력 값: {labels[0].item():.4f}")
        print(f"  🤖 예측된 응력 값: {prediction.item():.4f}")