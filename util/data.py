def pad_sequence(seqs, seq_length):
  features = []
  for seq in seqs:
    seq = seq + [1] * (seq_length - len(seq)) # 1 is the index of <pad>
    features.append(seq)
  return features
