# -transformer
1. embedding (+ positional encoding)
  - 단어를 정해진 차원으로 embedding 한다 ( ex: 512 차원이면 '1' 이 나타내는 단어는 1x512 차원의 벡터가 됨)
  - positional encoding 을 통해 위치값을 더해줌 (sinosidal positional encoding)
2. Multi-head-attention
  - head 개수만큼 나눠서 attention 후 다시 concat 한다 (d_k = d_v = d_model/h 로 논문에선 사용)
  - attention 은 보통 scaled dot product attention
3. Masked M.H.A
  - i 번째 출력이 i+1번째 input으로 들어가므로 masking out된 scaled dot-product attention 사용
  - masking out은 i 번째 위치에 대한 attention을 얻을 시 그 이후의 모든 위치에 대해선 attention softmax input이 -무한대가 되도록 한 
